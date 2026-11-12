-- gamestate_dataset_cv/src/ui.lua
-- Contains SMODS settings page (config_tab), in-game calibration overlay, and keypressed handlers.

local UI = {}

function G.DATASET_MOD:get_selected_object()
    local t = self.selected_target or 'roi_joker_bar'
    if t == 'roi_joker_bar' then return self.config.rois.joker_bar, 'roi', "ROI: Joker Bar (Top)"
    elseif t == 'roi_shop' then return self.config.rois.shop, 'roi', "ROI: Shop Items (Jokers/Cards)"
    elseif t == 'roi_voucher' then return self.config.rois.voucher, 'roi', "ROI: Voucher Shelf"
    elseif t == 'roi_booster' then return self.config.rois.booster, 'roi', "ROI: Booster Packs (Shelf)"
    elseif t == 'roi_pack' then return self.config.rois.pack, 'roi', "ROI: Booster Opening Screen"
    elseif t == 'roi_consumeables' then return self.config.rois.consumeables, 'roi', "ROI: Consumables (Top Right)"
    elseif t == 'roi_card_hand' then return self.config.rois.card_hand, 'roi', "ROI: Card Hand"
    elseif t == 'box_joker_bar' then return self.config.boxes.joker_bar_card, 'box', "Box: Joker Bar Card"
    elseif t == 'box_item' then return self.config.boxes.item_card, 'box', "Box: Shop/Pack Item"
    elseif t == 'box_consumeable' then return self.config.boxes.consumeable_card, 'box', "Box: Consumable Card"
    elseif t == 'box_voucher' then return self.config.boxes.voucher_card, 'box', "Box: Voucher Card"
    elseif t == 'box_booster' then return self.config.boxes.booster_card, 'box', "Box: Booster Pack"
    elseif t == 'box_card' then return self.config.boxes.card, 'box', "Box: Hand Card"
    end
    return nil, nil, "None"
end

-- Helper: Extract and sort all Joker centers with names and rarities
function UI.get_all_joker_entries()
    local entries = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Joker' and k ~= 'j_undiscovered' and k ~= 'j_locked' then
                local display_name = (v.name and v.name ~= "") and v.name or k
                if G.localization and G.localization.descriptions and G.localization.descriptions.Joker and G.localization.descriptions.Joker[k] then
                    local loc_entry = G.localization.descriptions.Joker[k]
                    if loc_entry.name then
                        if type(loc_entry.name) == "table" then
                            display_name = table.concat(loc_entry.name, " ")
                        else
                            display_name = tostring(loc_entry.name)
                        end
                    end
                elseif type(localize) == "function" then
                    local ok, loc_name = pcall(localize, {type = 'name_text', set = 'Joker', key = k})
                    if ok and loc_name and loc_name ~= "" then
                        display_name = loc_name
                    end
                end
                table.insert(entries, {
                    key = k,
                    name = display_name,
                    rarity = v.rarity or 1,
                    center = v
                })
            end
        end
        table.sort(entries, function(a, b)
            return (a.name or a.key):lower() < (b.name or b.key):lower()
        end)
    end
    return entries
end

function UI.get_filtered_joker_entries(mod)
    local all_entries = UI.get_all_joker_entries()
    local r_filter = mod.test_joker_rarity_filter or "All"
    if r_filter == "All" then
        return all_entries
    end
    local target_r = 1
    if r_filter == "Uncommon" then target_r = 2
    elseif r_filter == "Rare" then target_r = 3
    elseif r_filter == "Legendary" then target_r = 4
    end

    local filtered = {}
    for _, ent in ipairs(all_entries) do
        if ent.rarity == target_r then
            table.insert(filtered, ent)
        end
    end
    if #filtered == 0 then return all_entries end
    return filtered
end

function UI.spawn_test_joker(mod)
    local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
    local idx = mod.test_joker_idx or 1
    local selected = entries[idx]
    if not selected then
        mod.capture_flash = 1.5
        mod.capture_flash_text = "NO JOKER SELECTED!"
        return
    end

    if not G.jokers then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "NO ACTIVE RUN DETECTED! START A RUN FIRST."
        print("GDCV_JOKER_SPAWN: Failed - G.jokers is nil (not in a run)")
        return
    end

    -- Create card safely in G.jokers area
    local target_key = selected.key
    local card = create_card('Joker', G.jokers, nil, nil, nil, nil, target_key, 'deck')
    if not card then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "FAILED TO CREATE CARD: " .. tostring(target_key)
        return
    end

    -- Apply Edition
    local ed_choice = mod.test_joker_edition or "None"
    if ed_choice == "Foil" then
        card:set_edition({foil = true}, true, true)
    elseif ed_choice == "Holo" then
        card:set_edition({holo = true}, true, true)
    elseif ed_choice == "Polychrome" then
        card:set_edition({polychrome = true}, true, true)
    elseif ed_choice == "Negative" then
        card:set_edition({negative = true}, true, true)
    end

    -- Apply Sticker
    local st_choice = mod.test_joker_sticker or "None"
    if st_choice == "Eternal" and card.set_eternal then
        card:set_eternal(true)
    elseif st_choice == "Perishable" and card.set_perishable then
        card:set_perishable(true)
    elseif st_choice == "Rental" and card.set_rental then
        card:set_rental(true)
    end

    -- Register with active deck and board
    card:add_to_deck()
    G.jokers:emplace(card)

    -- Extra safety: expand card limit if slot capacity reached so test jokers always fit
    if G.jokers.config and G.jokers.config.card_limit and #G.jokers.cards > G.jokers.config.card_limit then
        G.jokers.config.card_limit = #G.jokers.cards
    end

    if play_sound then
        pcall(play_sound, 'card1')
    end
    if card.juice_up then
        pcall(function() card:juice_up(0.4, 0.4) end)
    end

    mod.capture_flash = 1.8
    mod.capture_flash_text = string.format("SPAWNED: %s (%s)", selected.name, target_key)
    print(string.format("GDCV_JOKER_SPAWN: Successfully spawned %s [%s] (Edition=%s, Sticker=%s)",
        selected.name, target_key, ed_choice, st_choice))
end

function UI.clear_all_jokers(mod)
    if not G.jokers or not G.jokers.cards or #G.jokers.cards == 0 then
        mod.capture_flash = 1.5
        mod.capture_flash_text = "NO JOKERS TO CLEAR!"
        return
    end

    local count = #G.jokers.cards
    for i = count, 1, -1 do
        local c = G.jokers.cards[i]
        if c then
            if c.remove_from_deck then
                pcall(function() c:remove_from_deck() end)
            end
            G.jokers:remove_card(c)
            if c.remove then
                pcall(function() c:remove() end)
            end
        end
    end

    mod.capture_flash = 1.8
    mod.capture_flash_text = string.format("REMOVED %d JOKERS FROM BOARD!", count)
    print(string.format("GDCV_JOKER_SPAWN: Cleared %d jokers from active joker bar", count))
end

function UI.setup_custom_test_suite(mod)
    if not G.jokers or not G.consumeables or not G.playing_cards then
        mod.capture_flash = 2.0
        mod.capture_flash_text = "START A RUN FIRST TO POPULATE TEST SUITE!"
        print("GDCV_TEST_SUITE: Failed - Not in an active run")
        return
    end

    -- 1. All custom Jokers for which the Python engine tracks state / hooks
    local custom_joker_keys = {
        -- Rotating targets
        "j_castle", "j_ancient", "j_mail", "j_idol", "j_todo_list",
        -- Scaling mult
        "j_green_joker", "j_ride_the_bus", "j_flash", "j_red_card",
        "j_trousers", "j_popcorn", "j_ceremonial",
        -- Scaling x_mult
        "j_campfire", "j_constellation", "j_hologram", "j_vampire", "j_obelisk",
        "j_yorick", "j_hit_the_road", "j_caino", "j_glass", "j_lucky_cat", "j_madness",
        -- Scaling chips & count-dependent chips
        "j_runner", "j_square", "j_wee", "j_ice_cream", "j_stone", "j_bull",
        "j_blue_joker", "j_stuntman", "j_steel_joker", "j_erosion",
        -- Sell value
        "j_egg", "j_gift", "j_swashbuckler",
        -- Spawners & card generation
        "j_riff_raff", "j_cartomancer", "j_vagabond", "j_sixth_sense", "j_seance",
        "j_superposition", "j_certificate", "j_marble", "j_hallucination", "j_8_ball",
        "j_dna", "j_trading", "j_perkeo",
        -- Mechanics & rules modifiers
        "j_midas_mask", "j_pareidolia", "j_oops", "j_four_fingers", "j_shortcut",
        "j_smeared", "j_splash", "j_mime", "j_blueprint", "j_brainstorm", "j_hack",
        "j_hanging_chad", "j_dusk", "j_sock_and_buskin", "j_burnt", "j_selzer",
        "j_turtle_bean", "j_ramen", "j_invisible", "j_burglar", "j_luchador",
        "j_mr_bones", "j_space", "j_business", "j_reserved_parking",
        "j_gros_michel", "j_cavendish", "j_cloud_9", "j_rocket", "j_satellite",
        "j_fortune_teller", "j_supernova", "j_throwback", "j_loyalty_card",
        "j_card_sharp", "j_to_the_moon", "j_astronomer", "j_diet_cola", "j_hiker",
        "j_photograph"
    }

    local owned_joker_keys = {}
    for _, j in ipairs(G.jokers.cards) do
        local k = j.config and j.config.center and j.config.center.key
        if k then owned_joker_keys[k] = true end
    end

    local jokers_spawned = 0
    for _, k in ipairs(custom_joker_keys) do
        if not owned_joker_keys[k] and G.P_CENTERS[k] then
            local card = create_card('Joker', G.jokers, nil, nil, true, nil, k, 'deck')
            if card then
                card:set_edition({negative = true}, true, true)
                card:add_to_deck()
                G.jokers:emplace(card)
                jokers_spawned = jokers_spawned + 1
            end
        end
    end
    G.jokers.config.card_limit = math.max(G.jokers.config.card_limit or 5, #G.jokers.cards)

    -- Ensure Ceremonial Dagger is moved to the far right so it doesn't destroy other Jokers on blind select
    local dagger_idx = nil
    if G.jokers and G.jokers.cards then
        for idx, j in ipairs(G.jokers.cards) do
            local k = j.config and j.config.center and j.config.center.key
            if k == "j_ceremonial" or (j.ability and j.ability.name == "Ceremonial Dagger") then
                dagger_idx = idx
                break
            end
        end
        if dagger_idx and dagger_idx < #G.jokers.cards then
            local dagger_card = table.remove(G.jokers.cards, dagger_idx)
            table.insert(G.jokers.cards, dagger_card)
            if G.jokers.set_ranks then G.jokers:set_ranks() end
            if G.jokers.align_cards then G.jokers:align_cards() end
        end
    end

    -- 2. Convert half of all cards in deck to Lucky Cards, half to Glass Cards
    local total_cards = #G.playing_cards
    local half_cards = math.floor(total_cards / 2)
    for i, c in ipairs(G.playing_cards) do
        if i <= half_cards then
            c:set_ability(G.P_CENTERS.m_lucky)
        else
            c:set_ability(G.P_CENTERS.m_glass)
        end
    end

    -- 3. Give Negative copies of all custom consumables (deck destruction / card manipulation)
    local custom_consumables = {
        -- Tarots
        "c_hanged_man", "c_death", "c_strength", "c_magician", "c_empress",
        "c_heirophant", "c_lovers", "c_chariot", "c_justice", "c_tower",
        "c_devil", "c_sun", "c_moon", "c_star", "c_world", "c_wheel_of_fortune",
        "c_fool", "c_hermit", "c_temperance",
        -- Spectrals
        "c_familiar", "c_grim", "c_incantation", "c_immolate", "c_cryptid",
        "c_ouija", "c_sigil", "c_aura", "c_deja_vu", "c_trance", "c_medium",
        "c_talisman", "c_ankh", "c_hex", "c_ectoplasm"
    }

    local owned_consumable_keys = {}
    for _, c in ipairs(G.consumeables.cards) do
        local k = c.config and c.config.center and c.config.center.key
        if k then owned_consumable_keys[k] = true end
    end

    local consumables_spawned = 0
    for _, k in ipairs(custom_consumables) do
        if not owned_consumable_keys[k] and G.P_CENTERS[k] then
            local c_set = (G.P_CENTERS[k] and G.P_CENTERS[k].set) or "Tarot"
            local card = create_card(c_set, G.consumeables, nil, nil, true, nil, k, 'deck')
            if card then
                card:set_edition({negative = true}, true, true)
                card:add_to_deck()
                G.consumeables:emplace(card)
                consumables_spawned = consumables_spawned + 1
            end
        end
    end
    G.consumeables.config.card_limit = math.max(G.consumeables.config.card_limit or 2, #G.consumeables.cards)

    if play_sound then pcall(play_sound, 'card1') end
    mod.capture_flash = 2.5
    mod.capture_flash_text = string.format("TEST SUITE: +%d JOKERS, +%d CONSUMABLES, %d CARDS 50/50 LUCKY/GLASS!",
        jokers_spawned, consumables_spawned, total_cards)
    print(string.format("GDCV_TEST_SUITE: Spawned %d Jokers (Negative), %d Consumables (Negative), Ceremonial Dagger on far right, converted %d deck cards (half Lucky / half Glass)",
        jokers_spawned, consumables_spawned, total_cards))

    if G.DATASET_MOD and G.DATASET_MOD.RunTracker and G.DATASET_MOD.RunTracker.snapshot_gamestate then
        G.DATASET_MOD.RunTracker.snapshot_gamestate("Setup Test Suite", {
            jokers_spawned = jokers_spawned,
            consumables_spawned = consumables_spawned,
            total_cards = total_cards
        })
    end
end

-- ==========================================
-- SMODS CONFIG TAB & MENU CALLBACKS
-- ==========================================
function UI.init_callbacks(mod)
    G.FUNCS = G.FUNCS or {}

    G.FUNCS.gdcv_change_mode = function(args)
        local mode_map = {
            ['Off'] = 'off',
            ['Card Classifier'] = 'card_classifier',
            ['Shop Classifier'] = 'shop_classifier'
        }
        local chosen = mode_map[args.to_val] or 'off'
        mod:set_mode(chosen)
    end

    G.FUNCS.gdcv_change_calib_state = function(args)
        local state_map = {
            ['Setup Mode'] = 'setup',
            ['Generation Mode'] = 'generation'
        }
        mod.config.calibration_mode = state_map[args.to_val] or 'setup'
        mod.state = 'idle'
    end

    G.FUNCS.gdcv_change_max_frames = function(args)
        local frames = tonumber(args.to_val) or 10000
        mod.config.max_frames = frames
    end

    G.FUNCS.gdcv_change_perkeo_max = function(args)
        local max_c = tonumber(args.to_val) or 10
        mod.config.perkeo_max_consumables = max_c
    end

    G.FUNCS.gdcv_change_card_selection_chance = function(args)
        mod.config.card_selection_chance = args.to_val or '50%'
        print("DATASET: Card selection chance set to " .. tostring(mod.config.card_selection_chance))
    end

    G.FUNCS.gdcv_save_config = function(e)
        mod:save_config()
        mod.capture_flash = 1.5
        mod.capture_flash_text = "CONFIG SAVED TO BOXES.LUA!"
    end

    G.FUNCS.gdcv_load_config = function(e)
        mod:load_config()
        mod.capture_flash = 1.5
        mod.capture_flash_text = "CONFIG LOADED FROM BOXES.LUA!"
    end

    G.FUNCS.gdcv_clear_dataset = function(e)
        if not mod.clear_confirm then
            mod.clear_confirm = true
            mod.clear_confirm_timer = 5.0
            mod.clear_button_text = "CONFIRM CLEAR?"
            mod.capture_flash = 1.0
            mod.capture_flash_text = "CLICK AGAIN WITHIN 5s TO ERASE ALL FRAMES!"
        else
            mod.clear_confirm = false
            mod.clear_confirm_timer = 0
            local count = mod:clear_all_dataset_data()
            mod.clear_button_text = "DATASET CLEARED!"
            mod.clear_reset_timer = 2.5
            mod.capture_flash = 2.0
            mod.capture_flash_text = string.format("CLEARED %d DATASET FILES!", count)
        end
    end

    G.FUNCS.gdcv_reload_mod = function(e)
        if mod.reload then
            mod:reload()
        else
            local ok, err = pcall(function()
                local function load_sub(rel)
                    if SMODS and SMODS.load_file then
                        local c = SMODS.load_file(rel, "gamestate_dataset_cv") or SMODS.load_file(rel)
                        if c then return c() end
                    end
                    if love.filesystem.getInfo("Mods/gamestate_dataset_cv/" .. rel) then
                        local content = love.filesystem.read("Mods/gamestate_dataset_cv/" .. rel)
                        local c = (loadstring or load)(content)
                        if c then return c() end
                    end
                end
                load_sub("src/config.lua")
                load_sub("src/cards_gen.lua")
                load_sub("src/shop_gen.lua")
                load_sub("src/capture.lua")
                local ui_m = load_sub("src/ui.lua")
                if ui_m then
                    ui_m.init_callbacks(mod)
                    if SMODS and SMODS.current_mod then SMODS.current_mod.config_tab = ui_m.create_config_tab end
                    if SMODS and SMODS.Mods and SMODS.Mods["gamestate_dataset_cv"] then
                        SMODS.Mods["gamestate_dataset_cv"].config_tab = ui_m.create_config_tab
                    end
                end
            end)
            if ok then
                mod.capture_flash = 1.5
                mod.capture_flash_text = "MOD HOT-RELOADED!"
            else
                mod.capture_flash = 2.0
                mod.capture_flash_text = "RELOAD FAILED: " .. tostring(err):sub(1, 40)
            end
        end
    end

    G.FUNCS.gdcv_restart_game = function(e)
        if SMODS and SMODS.restart_game then
            SMODS.restart_game()
        else
            love.event.quit("restart")
        end
    end

    -- ==========================================
    -- JOKER TEST GENERATOR CALLBACKS
    -- ==========================================
    G.FUNCS.gdcv_change_joker_rarity = function(args)
        local r_name = args.to_val or "All"
        mod.test_joker_rarity_filter = r_name
        local entries = UI.get_filtered_joker_entries(mod)
        mod.test_joker_list = entries
        mod.test_joker_idx = 1
        local sel = entries[1]
        mod.test_joker_name_display = sel and sel.name or "None"
        mod.test_joker_key_display = sel and ("[" .. sel.key .. "]") or "[-]"
    end

    G.FUNCS.gdcv_change_joker_selection = function(args)
        local idx = args.to_key
        if not idx then
            -- Fallback by searching option value in list
            local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
            for i, ent in ipairs(entries) do
                if ent.name == args.to_val then
                    idx = i
                    break
                end
            end
        end
        idx = idx or 1
        mod.test_joker_idx = idx
        local entries = mod.test_joker_list or UI.get_filtered_joker_entries(mod)
        local sel = entries[idx]
        mod.test_joker_name_display = sel and sel.name or "None"
        mod.test_joker_key_display = sel and ("[" .. sel.key .. "]") or "[-]"
    end

    G.FUNCS.gdcv_change_joker_edition = function(args)
        mod.test_joker_edition = args.to_val or "None"
    end

    G.FUNCS.gdcv_change_joker_sticker = function(args)
        mod.test_joker_sticker = args.to_val or "None"
    end

    G.FUNCS.gdcv_spawn_selected_joker = function(e)
        UI.spawn_test_joker(mod)
    end

    G.FUNCS.gdcv_clear_all_jokers = function(e)
        UI.clear_all_jokers(mod)
    end

    G.FUNCS.gdcv_setup_custom_test_suite = function(e)
        UI.setup_custom_test_suite(mod)
    end
end

function UI.create_config_tab()
    local mod = G.DATASET_MOD
    mod.clear_confirm = false
    mod.clear_confirm_timer = 0
    mod.clear_button_text = "Clear Dataset"
    mod.stats_text = string.format("Card Frames: %d  |  Shop Frames: %d  |  Phase: %s",
        mod.frame_counts and mod.frame_counts.card or 0,
        mod.frame_counts and mod.frame_counts.shop or 0,
        string.upper(mod.shop_phase or 'shop'))

    local current_mode_idx = (mod.config.mode == 'card_classifier' and 2) or (mod.config.mode == 'shop_classifier' and 3) or 1

    local max_frames_options = {'1000', '2500', '5000', '10000', '20000'}
    local current_mf_str = tostring(mod.config.max_frames or 10000)
    local current_mf_idx = 4
    for idx, val in ipairs(max_frames_options) do
        if val == current_mf_str then current_mf_idx = idx; break end
    end

    local perkeo_max_options = {'4', '5', '6', '7', '8', '9', '10'}
    local current_pm_str = tostring(mod.config.perkeo_max_consumables or 10)
    local current_pm_idx = 7
    for idx, val in ipairs(perkeo_max_options) do
        if val == current_pm_str then current_pm_idx = idx; break end
    end

    local selection_chance_options = {'0%', '50%', '100%'}
    local current_sc_str = tostring(mod.config.card_selection_chance or '50%')
    local current_sc_idx = 2
    for idx, val in ipairs(selection_chance_options) do
        if val == current_sc_str then current_sc_idx = idx; break end
    end

    -- Initialize Joker Generator State
    mod.test_joker_rarity_filter = mod.test_joker_rarity_filter or "All"
    mod.test_joker_edition = mod.test_joker_edition or "None"
    mod.test_joker_sticker = mod.test_joker_sticker or "None"
    local joker_entries = UI.get_filtered_joker_entries(mod)
    mod.test_joker_list = joker_entries
    mod.test_joker_idx = math.min(mod.test_joker_idx or 1, math.max(1, #joker_entries))

    local joker_names = {}
    for _, ent in ipairs(joker_entries) do
        table.insert(joker_names, ent.name)
    end
    if #joker_names == 0 then
        joker_names = {"No Jokers Found"}
    end

    local sel_entry = joker_entries[mod.test_joker_idx]
    mod.test_joker_name_display = sel_entry and sel_entry.name or "None"
    mod.test_joker_key_display = sel_entry and ("[" .. sel_entry.key .. "]") or "[-]"

    local rarity_options = {"All", "Common", "Uncommon", "Rare", "Legendary"}
    local rarity_idx = 1
    for i, r in ipairs(rarity_options) do
        if r == mod.test_joker_rarity_filter then rarity_idx = i; break end
    end

    local edition_options = {"None", "Foil", "Holo", "Polychrome", "Negative"}
    local edition_idx = 1
    for i, ed in ipairs(edition_options) do
        if ed == mod.test_joker_edition then edition_idx = i; break end
    end

    local sticker_options = {"None", "Eternal", "Perishable", "Rental"}
    local sticker_idx = 1
    for i, st in ipairs(sticker_options) do
        if st == mod.test_joker_sticker then sticker_idx = i; break end
    end

    local tab_def = {
        n = G.UIT.ROOT,
        config = {
            r = 0.1,
            minw = 6.8,
            align = "cm",
            padding = 0.08,
            colour = G.C.BLACK
        },
        nodes = {
            -- Title
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.04},
                nodes = {
                    {n = G.UIT.T, config = {text = "GameState Dataset CV - Settings", scale = 0.42, colour = G.C.GOLD}}
                }
            },
            -- Telemetry Engine Status
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.02},
                nodes = {
                    {
                        n = G.UIT.C,
                        config = {align = "cm", padding = 0.04, r = 0.1, colour = G.C.L_BLACK, minw = 6.4},
                        nodes = {
                            {
                                n = G.UIT.T,
                                config = {
                                    text = "Engine Parity Tracker: AUTO-ACTIVE (Recording to run_telemetry.jsonl)",
                                    scale = 0.28,
                                    colour = G.C.GREEN
                                }
                            }
                        }
                    }
                }
            },
            -- Mode Selector Cycle
            {
                n = G.UIT.R,
                config = {align = "cr", padding = 0.02},
                nodes = {
                    {n = G.UIT.C, config = {align = "cl", minw = 3.0}, nodes = {
                        {n = G.UIT.T, config = {text = "Classifier Mode", scale = 0.35, colour = G.C.UI.TEXT_LIGHT}}
                    }},
                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                        create_option_cycle({
                            id = 'gdcv_mode_cycle',
                            options = {'Off', 'Card Classifier', 'Shop Classifier'},
                            current_option = current_mode_idx,
                            opt_callback = 'gdcv_change_mode',
                            w = 2.8,
                            h = 0.6,
                            scale = 0.65
                        })
                    }}
                }
            },
            -- Target Frames Cap Cycle
            {
                n = G.UIT.R,
                config = {align = "cr", padding = 0.02},
                nodes = {
                    {n = G.UIT.C, config = {align = "cl", minw = 3.0}, nodes = {
                        {n = G.UIT.T, config = {text = "Target Frame Cap", scale = 0.35, colour = G.C.UI.TEXT_LIGHT}}
                    }},
                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                        create_option_cycle({
                            id = 'gdcv_mf_cycle',
                            options = max_frames_options,
                            current_option = current_mf_idx,
                            opt_callback = 'gdcv_change_max_frames',
                            w = 2.8,
                            h = 0.6,
                            scale = 0.65
                        })
                    }}
                }
            },
            -- Perkeo Max Consumables Cycle
            {
                n = G.UIT.R,
                config = {align = "cr", padding = 0.02},
                nodes = {
                    {n = G.UIT.C, config = {align = "cl", minw = 3.0}, nodes = {
                        {n = G.UIT.T, config = {text = "Perkeo Max Consumables", scale = 0.35, colour = G.C.UI.TEXT_LIGHT}}
                    }},
                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                        create_option_cycle({
                            id = 'gdcv_pm_cycle',
                            options = perkeo_max_options,
                            current_option = current_pm_idx,
                            opt_callback = 'gdcv_change_perkeo_max',
                            w = 2.8,
                            h = 0.6,
                            scale = 0.65
                        })
                    }}
                }
            },
            -- Card Selection Chance Cycle
            {
                n = G.UIT.R,
                config = {align = "cr", padding = 0.02},
                nodes = {
                    {n = G.UIT.C, config = {align = "cl", minw = 3.0}, nodes = {
                        {n = G.UIT.T, config = {text = "Card Selection Chance", scale = 0.35, colour = G.C.UI.TEXT_LIGHT}}
                    }},
                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                        create_option_cycle({
                            id = 'gdcv_sc_cycle',
                            options = selection_chance_options,
                            current_option = current_sc_idx,
                            opt_callback = 'gdcv_change_card_selection_chance',
                            w = 2.8,
                            h = 0.6,
                            scale = 0.65
                        })
                    }}
                }
            },
            -- =========================================================
            -- JOKER SPAWNER TEST BENCH (RNG & GameState Re-creation)
            -- =========================================================
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {
                        n = G.UIT.C,
                        config = {align = "cm", padding = 0.06, r = 0.1, colour = G.C.L_BLACK, minw = 6.5},
                        nodes = {
                            -- Header
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0.02},
                                nodes = {
                                    {
                                        n = G.UIT.T,
                                        config = {
                                            text = "JOKER TEST GENERATOR (GameState / RNG)",
                                            scale = 0.34,
                                            colour = G.C.ORANGE
                                        }
                                    }
                                }
                            },
                            -- Rarity Filter Cycle
                            {
                                n = G.UIT.R,
                                config = {align = "cr", padding = 0.02},
                                nodes = {
                                    {n = G.UIT.C, config = {align = "cl", minw = 2.8}, nodes = {
                                        {n = G.UIT.T, config = {text = "Rarity Filter", scale = 0.30, colour = G.C.UI.TEXT_LIGHT}}
                                    }},
                                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                                        create_option_cycle({
                                            id = 'gdcv_joker_rarity_cycle',
                                            options = rarity_options,
                                            current_option = rarity_idx,
                                            opt_callback = 'gdcv_change_joker_rarity',
                                            w = 3.2,
                                            h = 0.55,
                                            scale = 0.60
                                        })
                                    }}
                                }
                            },
                            -- Joker Selector Cycle
                            {
                                n = G.UIT.R,
                                config = {align = "cr", padding = 0.02},
                                nodes = {
                                    {n = G.UIT.C, config = {align = "cl", minw = 2.8}, nodes = {
                                        {n = G.UIT.T, config = {text = "Select Joker", scale = 0.30, colour = G.C.UI.TEXT_LIGHT}}
                                    }},
                                    {n = G.UIT.C, config = {align = "cm"}, nodes = {
                                        create_option_cycle({
                                            id = 'gdcv_joker_choice_cycle',
                                            options = joker_names,
                                            current_option = math.min(mod.test_joker_idx, #joker_names),
                                            opt_callback = 'gdcv_change_joker_selection',
                                            w = 3.2,
                                            h = 0.55,
                                            scale = 0.55
                                        })
                                    }}
                                }
                            },
                            -- Active Key Display
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0.01},
                                nodes = {
                                    {
                                        n = G.UIT.T,
                                        config = {
                                            ref_table = mod,
                                            ref_value = "test_joker_key_display",
                                            text = mod.test_joker_key_display,
                                            scale = 0.26,
                                            colour = G.C.BLUE
                                        }
                                    }
                                }
                            },
                            -- Modifiers: Edition & Sticker Cycle Row
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0.02},
                                nodes = {
                                    {n = G.UIT.C, config = {align = "cm", padding = 0.01}, nodes = {
                                        {n = G.UIT.T, config = {text = "Edition:", scale = 0.28, colour = G.C.UI.TEXT_LIGHT}},
                                        create_option_cycle({
                                            id = 'gdcv_joker_edition_cycle',
                                            options = edition_options,
                                            current_option = edition_idx,
                                            opt_callback = 'gdcv_change_joker_edition',
                                            w = 2.2,
                                            h = 0.5,
                                            scale = 0.55
                                        })
                                    }},
                                    {n = G.UIT.B, config = {w = 0.15, h = 0.1}},
                                    {n = G.UIT.C, config = {align = "cm", padding = 0.01}, nodes = {
                                        {n = G.UIT.T, config = {text = "Sticker:", scale = 0.28, colour = G.C.UI.TEXT_LIGHT}},
                                        create_option_cycle({
                                            id = 'gdcv_joker_sticker_cycle',
                                            options = sticker_options,
                                            current_option = sticker_idx,
                                            opt_callback = 'gdcv_change_joker_sticker',
                                            w = 2.2,
                                            h = 0.5,
                                            scale = 0.55
                                        })
                                    }}
                                }
                            },
                            -- Action Buttons: Spawn Joker & Clear All Jokers
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0.03},
                                nodes = {
                                    UIBox_button({
                                        button = 'gdcv_spawn_selected_joker',
                                        label = {'+ Spawn Joker'},
                                        minw = 2.9,
                                        minh = 0.55,
                                        scale = 0.32,
                                        colour = G.C.GOLD
                                    }),
                                    {n = G.UIT.B, config = {w = 0.2, h = 0.1}},
                                    UIBox_button({
                                        button = 'gdcv_clear_all_jokers',
                                        label = {'Clear Jokers'},
                                        minw = 2.9,
                                        minh = 0.55,
                                        scale = 0.32,
                                        colour = G.C.RED
                                    })
                                }
                            },
                            -- Custom Test Suite Setup: Spawn All Custom Jokers & Consumables, Split Deck Lucky/Glass
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0.03},
                                nodes = {
                                    UIBox_button({
                                        button = 'gdcv_setup_custom_test_suite',
                                        label = {'Setup Custom Engine Test Suite'},
                                        minw = 6.0,
                                        minh = 0.55,
                                        scale = 0.32,
                                        colour = G.C.PURPLE
                                    })
                                }
                            }
                        }
                    }
                }
            },
            -- Stats & Info Row
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {
                        n = G.UIT.T,
                        config = {
                            ref_table = mod,
                            ref_value = "stats_text",
                            text = mod.stats_text,
                            scale = 0.30,
                            colour = G.C.UI.TEXT_INACTIVE
                        }
                    }
                }
            },
            -- Buttons Row 1: Save, Load, Hot Reload
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    UIBox_button({
                        button = 'gdcv_save_config',
                        label = {'Save Config'},
                        minw = 1.9,
                        minh = 0.55,
                        scale = 0.32,
                        colour = G.C.GREEN
                    }),
                    {n = G.UIT.B, config = {w = 0.15, h = 0.1}},
                    UIBox_button({
                        button = 'gdcv_load_config',
                        label = {'Load Config'},
                        minw = 1.9,
                        minh = 0.55,
                        scale = 0.32,
                        colour = G.C.ORANGE
                    }),
                    {n = G.UIT.B, config = {w = 0.15, h = 0.1}},
                    UIBox_button({
                        button = 'gdcv_reload_mod',
                        label = {'Hot Reload'},
                        minw = 1.9,
                        minh = 0.55,
                        scale = 0.32,
                        colour = G.C.BLUE
                    })
                }
            },
            -- Buttons Row 2: Clear Dataset (Confirmation) & Restart Game
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.03},
                nodes = {
                    {
                        n = G.UIT.C,
                        config = {
                            align = "cm",
                            padding = 0.05,
                            r = 0.1,
                            hover = true,
                            colour = G.C.RED,
                            button = 'gdcv_clear_dataset',
                            minw = 2.9,
                            minh = 0.55,
                            shadow = true,
                        },
                        nodes = {
                            {
                                n = G.UIT.R,
                                config = {align = "cm", padding = 0},
                                nodes = {
                                    {
                                        n = G.UIT.T,
                                        config = {
                                            ref_table = mod,
                                            ref_value = "clear_button_text",
                                            text = mod.clear_button_text,
                                            scale = 0.32,
                                            colour = G.C.UI.TEXT_LIGHT,
                                            shadow = true
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {n = G.UIT.B, config = {w = 0.2, h = 0.1}},
                    UIBox_button({
                        button = 'gdcv_restart_game',
                        label = {'Restart Game'},
                        minw = 2.9,
                        minh = 0.55,
                        scale = 0.32,
                        colour = G.C.PURPLE
                    })
                }
            }
        }
    }
    return tab_def
end

-- ==========================================
-- IN-GAME HUD & CALIBRATION OVERLAY
-- ==========================================
function G.DATASET_MOD:draw_debug()
    if self.config.mode == 'off' then return end
    if self.config.calibration_mode ~= 'setup' then return end

    love.graphics.push('all')

    -- 1. Draw Active ROIs
    local function draw_roi_box(roi, name, is_selected, r, g, b)
        if not roi then return end
        love.graphics.setColor(r, g, b, is_selected and 1.0 or 0.5)
        love.graphics.setLineWidth(is_selected and 4 or 2)
        love.graphics.rectangle('line', roi.x, roi.y, roi.w, roi.h)

        love.graphics.setColor(r, g, b, 0.9)
        love.graphics.print(name, roi.x + 5, roi.y - 18)
    end

    if self.config.mode == 'card_classifier' then
        local is_sel = (self.selected_target == 'roi_card_hand')
        draw_roi_box(self.config.rois.card_hand, "[ROI: CARD HAND]", is_sel, 0, 1, 0)
    elseif self.config.mode == 'shop_classifier' then
        local sel = self.selected_target
        draw_roi_box(self.config.rois.joker_bar, "[ROI: JOKER BAR (TOP)]", sel == 'roi_joker_bar', 0.2, 0.8, 1)
        draw_roi_box(self.config.rois.consumeables, "[ROI: CONSUMABLES (TOP RIGHT)]", sel == 'roi_consumeables', 1.0, 0.4, 0.9)

        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_shop' then
            draw_roi_box(self.config.rois.shop, "[ROI: SHOP ITEMS]", sel == 'roi_shop', 1, 0.8, 0.2)
        end
        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_voucher' then
            draw_roi_box(self.config.rois.voucher, "[ROI: VOUCHER SHELF]", sel == 'roi_voucher', 0.3, 1.0, 0.4)
        end
        if (self.shop_phase or 'shop') == 'shop' or sel == 'roi_booster' then
            draw_roi_box(self.config.rois.booster, "[ROI: BOOSTER PACKS (SHELF)]", sel == 'roi_booster', 0.9, 0.6, 0.2)
        end
        if (self.shop_phase or 'shop') == 'pack' or sel == 'roi_pack' then
            local kind_name = (self.current_pack_kind and self.current_pack_kind ~= 'Pack' and self.current_pack_kind) or 'BOOSTER'
            local p_name = string.format("[ROI: %s OPENING SCREEN]", string.upper(kind_name))
            draw_roi_box(self.config.rois.pack, p_name, sel == 'roi_pack', 1, 0.3, 0.8)
        end
    end

    -- 2. Draw Card Bounding Boxes Polygons
    local function is_card_active(card)
        if not card or card.REMOVED then return false end
        if self.config.mode == 'card_classifier' then
            if G.hand and G.hand.cards then
                for _, c in ipairs(G.hand.cards) do
                    if c == card then return true end
                end
            end
        elseif self.config.mode == 'shop_classifier' then
            if G.jokers and G.jokers.cards then
                for _, c in ipairs(G.jokers.cards) do
                    if c == card then return true end
                end
            end
            if (self.shop_phase or 'shop') == 'shop' then
                if G.shop_jokers and G.shop_jokers.cards then
                    for _, c in ipairs(G.shop_jokers.cards) do
                        if c == card then return true end
                    end
                end
                if G.shop_vouchers and G.shop_vouchers.cards then
                    for _, c in ipairs(G.shop_vouchers.cards) do
                        if c == card then return true end
                    end
                end
                if G.shop_booster and G.shop_booster.cards then
                    for _, c in ipairs(G.shop_booster.cards) do
                        if c == card then return true end
                    end
                end
            else
                if G.pack_cards and G.pack_cards.cards then
                    for _, c in ipairs(G.pack_cards.cards) do
                        if c == card then return true end
                    end
                end
            end
            if G.consumeables and G.consumeables.cards then
                for _, c in ipairs(G.consumeables.cards) do
                    if c == card then return true end
                end
            end
        end
        return false
    end

    if self.captured_boxes then
        for card, box in pairs(self.captured_boxes) do
            if not is_card_active(card) then
                self.captured_boxes[card] = nil
            elseif box and box.points then
                local pts = box.points
                local is_sel_box = (self.selected_target == 'box_joker_bar' and box.box_name == 'joker_bar_card') or
                                   (self.selected_target == 'box_item' and box.box_name == 'item_card') or
                                   (self.selected_target == 'box_consumeable' and box.box_name == 'consumeable_card') or
                                   (self.selected_target == 'box_voucher' and box.box_name == 'voucher_card') or
                                   (self.selected_target == 'box_booster' and box.box_name == 'booster_card') or
                                   (self.selected_target == 'box_card' and box.box_name == 'card')

                if box.box_name == 'joker_bar_card' then
                    love.graphics.setColor(0, 1, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'consumeable_card' then
                    love.graphics.setColor(1, 0.4, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'item_card' then
                    love.graphics.setColor(1, 1, 0, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'voucher_card' then
                    love.graphics.setColor(0.3, 1, 0.3, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'booster_card' then
                    love.graphics.setColor(1, 0.6, 0.1, is_sel_box and 1.0 or 0.6)
                else
                    love.graphics.setColor(0, 1, 1, is_sel_box and 1.0 or 0.6)
                end

                love.graphics.setLineWidth(is_sel_box and 3 or 1.5)
                if #pts >= 8 then
                    love.graphics.polygon('line', pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], pts[8])
                end
            end
        end
    end

    -- 3. Draw HUD Text Panel
    local _, obj_type, obj_label = self:get_selected_object()
    local obj = select(1, self:get_selected_object())

    love.graphics.setColor(0, 0, 0, 0.8)
    love.graphics.rectangle('fill', 8, 8, 920, 185, 8, 8)
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.setLineWidth(1)
    love.graphics.rectangle('line', 8, 8, 920, 185, 8, 8)

    local mode_str = (self.config.mode == 'card_classifier' and "CARD CLASSIFIER") or "SHOP & JOKER CLASSIFIER"
    love.graphics.setColor(1, 0.85, 0.2, 1)
    love.graphics.print(string.format("GAMESTATE CV CALIBRATION  -  [%s]", mode_str), 18, 14)

    love.graphics.setColor(1, 1, 1, 1)
    local coord_str = ""
    if obj and obj_type == 'roi' then
        coord_str = string.format("X: %.1f, Y: %.1f, W: %.1f, H: %.1f", obj.x, obj.y, obj.w, obj.h)
    elseif obj and obj_type == 'box' then
        coord_str = string.format("W: %.3f, H: %.3f, OX: %.3f, OY: %.3f", obj.w, obj.h, obj.ox, obj.oy)
    end
    love.graphics.print(string.format("Editing Target: %s  [%s]", obj_label, coord_str), 18, 36)

    love.graphics.setColor(0.3, 0.9, 1, 1)
    if self.config.mode == 'shop_classifier' then
        if (self.shop_phase or 'shop') == 'pack' then
            love.graphics.print("Keys: 1: ROI Jokers | 3/0: ROI Pack Open | 4: ROI Consumables | 5: Box Joker | 6: Box Item", 18, 58)
        else
            love.graphics.print("Keys: 1: ROI Jokers | 2: ROI Shop | 3: ROI Boosters | 4: ROI Consumables | V: ROI Vouchers | 5-9: Boxes", 18, 58)
        end
    else
        love.graphics.print("Keys: 1: ROI Card Hand | 2: Box Hand Card", 18, 58)
    end

    love.graphics.setColor(0.9, 0.9, 0.9, 1)
    love.graphics.print("W, A, S, D: Move / Shift Offset  |  Arrow Keys: Adjust Size (Hold Shift: Fast)", 18, 80)
    love.graphics.print("TAB: Toggle Gen/Setup  |  R: Randomize  |  P: Switch Shop/Pack View  |  C: Capture Test", 18, 102)
    love.graphics.print("[, ]: Save / Load Config (dataset/boxes.lua)", 18, 124)

    local cur_frame = (self.config.mode == 'shop_classifier') and self.frame_counts.shop or self.frame_counts.card
    local phase_extra = ""
    if self.config.mode == 'shop_classifier' then
        local p_kind = (self.shop_phase == 'pack') and (self.current_pack_kind or 'Pack') or "Shop"
        local perk_str = self.is_perkeo_active and " [PERKEO]" or ""
        local show_str = self.pack_has_showman and " [SHOWMAN]" or ""
        phase_extra = string.format(" | Phase: %s (%d/100)%s%s", p_kind, self.phase_step or 0, perk_str, show_str)
    end

    love.graphics.setColor(1, 1, 0.4, 1)
    love.graphics.print(string.format("Next Frame: %04d / %d%s | Lua Mem: %.1f MB", cur_frame, self.config.max_frames or 10000, phase_extra, collectgarbage("count") / 1024), 18, 146)

    local stg_cnt = (G.STAGE_OBJECTS and G.STAGE_OBJECTS[G.STAGE] and #G.STAGE_OBJECTS[G.STAGE]) or 0
    local mov_cnt = (G.MOVEABLES and #G.MOVEABLES) or 0
    local crd_cnt = (G.I and G.I.CARD and #G.I.CARD) or 0
    local spr_cnt = (G.I and G.I.SPRITE and #G.I.SPRITE) or 0
    local uib_cnt = (G.I and G.I.UIBOX and #G.I.UIBOX) or 0
    love.graphics.setColor(0.6, 0.8, 1, 0.8)
    love.graphics.print(string.format("Nodes: %d | Moveables: %d | Cards: %d | Sprites: %d | UIBoxes: %d", stg_cnt, mov_cnt, crd_cnt, spr_cnt, uib_cnt), 18, 168)

    -- Flash message
    if self.capture_flash and self.capture_flash > 0 then
        love.graphics.setColor(0, 1, 0, math.min(1.0, self.capture_flash))
        love.graphics.print(self.capture_flash_text or "SAVED!", 18, 196)
    end

    love.graphics.pop()
end

-- ==========================================
-- KEYPRESS HANDLER
-- ==========================================
function G.DATASET_MOD:on_keypressed(key)
    if self.config.mode == 'off' then return end

    if key == 'tab' then
        if self.config.calibration_mode == 'setup' then
            self.config.calibration_mode = 'generation'
            self.state = 'idle'
            self.capture_flash = 2.0
            self.capture_flash_text = "AUTO-GENERATION MODE STARTED!"
            print("DATASET: Generation Mode Started")
        else
            self.config.calibration_mode = 'setup'
            self.state = 'idle'
            self.capture_flash = 2.0
            self.capture_flash_text = "CALIBRATION SETUP MODE ACTIVE."
            print("DATASET: Setup Mode Started")
        end
    elseif self.config.calibration_mode == 'setup' then
        -- Target selections
        if self.config.mode == 'card_classifier' then
            if key == '1' then self.selected_target = 'roi_card_hand'
            elseif key == '2' then self.selected_target = 'box_card'
            end
        elseif self.config.mode == 'shop_classifier' then
            if key == '1' then self.selected_target = 'roi_joker_bar'
            elseif key == '2' then self.selected_target = 'roi_shop'
            elseif key == '3' then
                if self.shop_phase == 'pack' then
                    self.selected_target = 'roi_pack'
                else
                    self.selected_target = 'roi_booster'
                end
            elseif key == '4' then self.selected_target = 'roi_consumeables'
            elseif key == 'v' then self.selected_target = 'roi_voucher'
            elseif key == '0' then self.selected_target = 'roi_pack'
            elseif key == '5' then self.selected_target = 'box_joker_bar'
            elseif key == '6' then self.selected_target = 'box_item'
            elseif key == '7' then self.selected_target = 'box_consumeable'
            elseif key == '8' then self.selected_target = 'box_voucher'
            elseif key == '9' then self.selected_target = 'box_booster'
            end
        end

        -- Actions
        if key == 'r' then
            if self.config.mode == 'card_classifier' then
                self:generate_random_hand()
            else
                if (self.shop_phase or 'shop') == 'shop' then
                    self:setup_shop_screen()
                else
                    self:setup_pack_screen()
                end
            end
            if self.ShopGen and self.ShopGen.cleanup_cycle then
                self.ShopGen.cleanup_cycle()
            else
                collectgarbage("collect")
                collectgarbage("collect")
            end
        elseif key == 'p' and self.config.mode == 'shop_classifier' then
            self:toggle_shop_screen_mode()
        elseif key == 't' and self.config.mode == 'card_classifier' then
            self:randomize_hand_selection()
        elseif key == 'c' then
            self:capture_frame()
        elseif key == ',' or key == '[' then
            self:save_config()
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG SAVED!"
        elseif key == '.' or key == ']' then
            self:load_config()
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG LOADED!"
        end
    end
end

-- ==========================================
-- UPDATE HOOK (NUDGING & GENERATION)
-- ==========================================
function G.DATASET_MOD:update(dt)
    if self.capture_flash and self.capture_flash > 0 then
        self.capture_flash = math.max(0, self.capture_flash - dt)
    end

    if self.clear_confirm then
        self.clear_confirm_timer = (self.clear_confirm_timer or 0) - dt
        if self.clear_confirm_timer <= 0 then
            self.clear_confirm = false
            self.clear_button_text = "Clear Dataset"
        end
    elseif self.clear_reset_timer then
        self.clear_reset_timer = self.clear_reset_timer - dt
        if self.clear_reset_timer <= 0 then
            self.clear_reset_timer = nil
            self.clear_button_text = "Clear Dataset"
        end
    end

    self.stats_text = string.format("Card Frames: %d  |  Shop Frames: %d  |  Phase: %s",
        self.frame_counts and self.frame_counts.card or 0,
        self.frame_counts and self.frame_counts.shop or 0,
        string.upper(self.shop_phase or 'shop'))

    if not self.config or self.config.mode == 'off' then return end

    if self.config.calibration_mode == 'setup' then
        local obj, obj_type = self:get_selected_object()
        if not obj then return end

        if obj_type == 'roi' then
            local roi_step = love.keyboard.isDown('lshift', 'rshift') and (2500 * dt) or (750 * dt)
            if love.keyboard.isDown('w') then obj.y = obj.y - roi_step end
            if love.keyboard.isDown('s') then obj.y = obj.y + roi_step end
            if love.keyboard.isDown('a') then obj.x = obj.x - roi_step end
            if love.keyboard.isDown('d') then obj.x = obj.x + roi_step end
            if love.keyboard.isDown('up') then obj.h = obj.h - roi_step end
            if love.keyboard.isDown('down') then obj.h = obj.h + roi_step end
            if love.keyboard.isDown('left') then obj.w = obj.w - roi_step end
            if love.keyboard.isDown('right') then obj.w = obj.w + roi_step end
        elseif obj_type == 'box' then
            local step = love.keyboard.isDown('lshift', 'rshift') and (2.0 * dt) or (0.5 * dt)
            local moved = false
            if love.keyboard.isDown('w') then obj.oy = obj.oy - step; moved = true end
            if love.keyboard.isDown('s') then obj.oy = obj.oy + step; moved = true end
            if love.keyboard.isDown('a') then obj.ox = obj.ox - step; moved = true end
            if love.keyboard.isDown('d') then obj.ox = obj.ox + step; moved = true end
            if love.keyboard.isDown('up') then obj.h = obj.h - step; moved = true end
            if love.keyboard.isDown('down') then obj.h = obj.h + step; moved = true end
            if love.keyboard.isDown('left') then obj.w = obj.w - step; moved = true end
            if love.keyboard.isDown('right') then obj.w = obj.w + step; moved = true end
            if moved then
                self.captured_boxes = setmetatable({}, {__mode = "k"})
            end
        end
    elseif self.config.calibration_mode == 'generation' then
        local cur_frame = (self.config.mode == 'shop_classifier') and self.frame_counts.shop or self.frame_counts.card
        local max_frames = self.config.max_frames or 10000

        if cur_frame >= max_frames then
            self.config.calibration_mode = 'setup'
            self.state = 'idle'
            self.capture_flash = 3.0
            self.capture_flash_text = string.format("REACHED TARGET CAP (%d FRAMES)! GENERATION STOPPED.", max_frames)
            print(string.format("DATASET: Reached threshold cap of %d frames. Stopped generation.", max_frames))
            return
        end

        if self.state == 'idle' then
            if self.config.mode == 'card_classifier' then
                self:generate_random_hand()
                self:randomize_background()
            elseif self.config.mode == 'shop_classifier' then
                self:generate_shop_step()
            end

            self.state = 'waiting'
            self.timer = 0.08 -- Reduced from 0.25 since animations are instantly snapped
        elseif self.state == 'waiting' then
            self.timer = self.timer - dt
            if self.timer <= 0 then
                self.state = 'capturing'
                self:capture_frame()
            end
        end
    end
end

return UI
