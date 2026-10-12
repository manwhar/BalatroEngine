-- gamestate_dataset_cv/src/shop_gen.lua
-- Contains shop screen, booster pack, joker bar, and top consumable area generation.

local ShopGen = {}

-- Cached Center Pools
ShopGen.cached_jokers = nil
ShopGen.cached_non_legendary_jokers = nil
ShopGen.cached_tarots = nil
ShopGen.cached_spectrals = nil
ShopGen.cached_planets = nil

function ShopGen.get_all_jokers(allow_legendary)
    if allow_legendary == nil then allow_legendary = true end
    if allow_legendary and ShopGen.cached_jokers and #ShopGen.cached_jokers > 0 then
        return ShopGen.cached_jokers
    elseif not allow_legendary and ShopGen.cached_non_legendary_jokers and #ShopGen.cached_non_legendary_jokers > 0 then
        return ShopGen.cached_non_legendary_jokers
    end

    local all_j = {}
    local non_leg_j = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == 'Joker' and k ~= 'j_undiscovered' then
                table.insert(all_j, v)
                if v.rarity ~= 4 then
                    table.insert(non_leg_j, v)
                end
            end
        end
        table.sort(all_j, function(a, b) return (a.key or "") < (b.key or "") end)
        table.sort(non_leg_j, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    ShopGen.cached_jokers = all_j
    ShopGen.cached_non_legendary_jokers = non_leg_j

    return allow_legendary and all_j or non_leg_j
end

function ShopGen.get_all_consumables(set_type, include_hidden)
    local list = {}
    if G.P_CENTERS then
        for k, v in pairs(G.P_CENTERS) do
            if v.set == set_type and k ~= 'c_undiscovered' then
                local is_hidden = (k == 'c_soul' or k == 'c_black_hole')
                if include_hidden or not is_hidden then
                    table.insert(list, v)
                end
            end
        end
        table.sort(list, function(a, b) return (a.key or "") < (b.key or "") end)
    end
    return list
end

-- Cached BUMod Variants Manifest
ShopGen.bu_variants = nil

function ShopGen.load_bu_variants()
    if ShopGen.bu_variants ~= nil then return ShopGen.bu_variants end
    local ok, manifest = pcall(function()
        if SMODS and SMODS.load_file then
            local chunk = SMODS.load_file("src/bu_variants_manifest.lua", "gamestate_dataset_cv")
            if chunk then return chunk() end
        end
        return nil
    end)
    if ok and manifest then
        ShopGen.bu_variants = manifest
    else
        ShopGen.bu_variants = {}
    end
    return ShopGen.bu_variants
end

-- Cache loaded Love2D Images for BU variants so we don't reload from disk repeatedly
ShopGen.loaded_images = {}
function ShopGen.get_variant_image(filename)
    if ShopGen.loaded_images[filename] then
        return ShopGen.loaded_images[filename]
    end
    local path = "Mods/gamestate_dataset_cv/assets/bu_variants/" .. filename
    if love.filesystem.getInfo(path) then
        local img_data = love.image.newImageData(path)
        local img = love.graphics.newImage(img_data)
        ShopGen.loaded_images[filename] = img
        if img_data and img_data.release then img_data:release() end
        return img
    end
    return nil
end

ShopGen.cached_quads = {}
function ShopGen.get_quad(x, y, w, h, sw, sh)
    local k = string.format("%d_%d_%d_%d_%d_%d", x, y, w, h, sw, sh)
    if not ShopGen.cached_quads[k] then
        ShopGen.cached_quads[k] = love.graphics.newQuad(x, y, w, h, sw, sh)
    end
    return ShopGen.cached_quads[k]
end

function G.DATASET_MOD:get_joker_pools()
    if not ShopGen.cached_tarots or #ShopGen.cached_tarots == 0 then
        ShopGen.cached_tarots = ShopGen.get_all_consumables('Tarot', false)
        ShopGen.cached_spectrals = ShopGen.get_all_consumables('Spectral', false)
        ShopGen.cached_planets = ShopGen.get_all_consumables('Planet', false)
    end
    return ShopGen.get_all_jokers(true), ShopGen.cached_tarots, ShopGen.cached_spectrals, ShopGen.cached_planets
end

-- Create a fully randomized Joker card
function G.DATASET_MOD:create_random_joker_card(area, forced_center, allow_legendary)
    if allow_legendary == nil then allow_legendary = true end
    local jokers = ShopGen.get_all_jokers(allow_legendary)
    if #jokers == 0 then return nil end
    local center = forced_center or jokers[math.random(#jokers)]

    local card = Card(
        area.T.x + area.T.w / 2, area.T.y,
        G.CARD_W, G.CARD_H,
        nil, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos
        }
    )

    -- Editions distribution:
    -- None (60%), Foil (10%), Holo (10%), Polychrome (10%), Negative (10%)
    local ed_roll = math.random()
    if ed_roll < 0.10 then
        card:set_edition({foil = true}, true, true)
    elseif ed_roll < 0.20 then
        card:set_edition({holo = true}, true, true)
    elseif ed_roll < 0.30 then
        card:set_edition({polychrome = true}, true, true)
    elseif ed_roll < 0.40 then
        card:set_edition({negative = true}, true, true)
    end

    -- Stickers distribution:
    -- Eternal (15%), Perishable (15%) [mutually exclusive]
    -- Rental (20%) [independent]
    local stick_roll = math.random()
    if stick_roll < 0.15 then
        if card.set_eternal then card:set_eternal(true) else card.ability.eternal = true end
    elseif stick_roll < 0.30 then
        if card.set_perishable then card:set_perishable(true) else card.ability.perishable = true end
    end

    if math.random() < 0.20 then
        if card.set_rental then card:set_rental(true) else card.ability.rental = true end
    end

    -- BUMod Variant Selection:
    -- Check if this Joker has historical BUMod sprite variations
    local bu_manifest = ShopGen.load_bu_variants()
    local c_key = center.key
    local variants = bu_manifest[c_key]
    -- Handle trans jokers which cover j_seance and j_banner
    if not variants and (c_key == 'j_seance' or c_key == 'j_banner') then
        variants = bu_manifest['jokers_trans']
    end

    card.custom_art_variant = "vanilla_or_active"
    if variants and #variants > 0 then
        -- Evenly disperse: 50% active/vanilla mod art, 50% split across historical BUMod variants
        if math.random() < 0.50 then
            local chosen_var = variants[math.random(#variants)]
            local img = ShopGen.get_variant_image(chosen_var.filename)
            if img and card.children and card.children.center then
                card.children.center.image = img
                -- Set quad to entire 142x190 texture (or 1st/2nd half if jokers_trans)
                local w, h = img:getDimensions()
                if c_key == 'j_banner' and w > 200 then
                    card.children.center.quad = ShopGen.get_quad(71 * 2, 0, 71 * 2, 95 * 2, w, h)
                else
                    card.children.center.quad = ShopGen.get_quad(0, 0, 71 * 2, 95 * 2, w, h)
                end
                card.custom_art_variant = string.format("bumod_v%d_%s", chosen_var.version or 1, chosen_var.commit or "hist")
            end
        end
    end

    return card
end

-- Create a consumable card (Tarot, Spectral, Planet)
function G.DATASET_MOD:create_random_consumable_card(area, set_type, forced_center)
    local _, tarots, spectrals, planets = self:get_joker_pools()
    local pool = (set_type == 'Tarot' and tarots) or (set_type == 'Spectral' and spectrals) or planets
    if not pool or #pool == 0 then return nil end
    local center = forced_center or pool[math.random(#pool)]

    local card = Card(
        area.T.x + area.T.w / 2, area.T.y,
        G.CARD_W, G.CARD_H,
        nil, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos
        }
    )
    return card
end

-- Helper to attach shop card price and buy tags synchronously without queuing delayed events in G.E_MANAGER
local function attach_shop_card_ui_sync(card)
    if not card then return end
    local cost_str = card.cost and tostring(card.cost) or "4"
    local t1 = {
        n=G.UIT.ROOT, config = {minw = 0.6, align = 'tm', colour = darken(G.C.BLACK, 0.2), shadow = true, r = 0.05, padding = 0.05, minh = 1}, nodes={
            {n=G.UIT.R, config={align = "cm", colour = lighten(G.C.BLACK, 0.1), r = 0.1, minw = 1, minh = 0.55, emboss = 0.05, padding = 0.03}, nodes={
                {n=G.UIT.T, config={text = "$" .. cost_str, colour = G.C.MONEY, scale = 0.4}}
            }}
        }
    }
    local t2 = {
        n=G.UIT.ROOT, config = {minw = 1.1, maxw = 1.3, padding = 0.1, align = 'bm', colour = G.C.GOLD, shadow = true, r = 0.08, minh = 0.94}, nodes={
            {n=G.UIT.T, config={text = "BUY", colour = G.C.WHITE, scale = 0.45}}
        }
    }
    card.children.price = UIBox{
        definition = t1,
        config = {align="tm", offset = {x=0, y=1.5}, major = card, bond = 'Weak', parent = card}
    }
    card.children.buy_button = UIBox{
        definition = t2,
        config = {align="bm", offset = {x=0, y=-0.3}, major = card, bond = 'Weak', parent = card}
    }
end

-- Helper to safely and completely sever all references to a Card and its child objects
local function fully_remove_card(card)
    if not card then return end
    if G.DATASET_MOD and G.DATASET_MOD.captured_boxes then
        G.DATASET_MOD.captured_boxes[card] = nil
    end
    if card.area then
        card:remove_from_area()
    end
    if card.children then
        for k, child in pairs(card.children) do
            if child then
                if child.remove then
                    pcall(function() child:remove() end)
                end
                card.children[k] = nil
            end
        end
    end
    pcall(function() card:remove() end)
    card.children = nil
end
ShopGen.fully_remove_card = fully_remove_card

-- Helper to safely remove all cards from a CardArea and clean up their bounding boxes
local function clear_card_area(area)
    if area and area.cards then
        for i = #area.cards, 1, -1 do
            local c = area.cards[i]
            fully_remove_card(c)
        end
        area.cards = {}
    end
    if area and area.highlighted then
        area.highlighted = {}
    end
end
ShopGen.clear_card_area = clear_card_area

-- Sweep through all engine tracking tables and remove any dead objects
local function prune_engine_tables()
    if G.MOVEABLES then
        for i = #G.MOVEABLES, 1, -1 do
            if G.MOVEABLES[i] and G.MOVEABLES[i].REMOVED then
                table.remove(G.MOVEABLES, i)
            end
        end
    end
    if G.I then
        for _, tbl in pairs(G.I) do
            if type(tbl) == 'table' then
                for i = #tbl, 1, -1 do
                    if tbl[i] and tbl[i].REMOVED then
                        table.remove(tbl, i)
                    end
                end
            end
        end
    end
    if G.STAGE_OBJECTS then
        for _, stage_tbl in pairs(G.STAGE_OBJECTS) do
            if type(stage_tbl) == 'table' then
                for i = #stage_tbl, 1, -1 do
                    if stage_tbl[i] and stage_tbl[i].REMOVED then
                        table.remove(stage_tbl, i)
                    end
                end
            end
        end
    end
    if G.ANIMATIONS then
        for i = #G.ANIMATIONS, 1, -1 do
            if G.ANIMATIONS[i] and G.ANIMATIONS[i].REMOVED then
                table.remove(G.ANIMATIONS, i)
            end
        end
    end
end
ShopGen.prune_engine_tables = prune_engine_tables

-- Clear any lingering controller targets pointing to removed objects
local function clear_controller_targets()
    if G.CONTROLLER then
        if G.CONTROLLER.hovering then
            if G.CONTROLLER.hovering.target and G.CONTROLLER.hovering.target.REMOVED then
                G.CONTROLLER.hovering.target = nil
            end
            if G.CONTROLLER.hovering.prev_target and G.CONTROLLER.hovering.prev_target.REMOVED then
                G.CONTROLLER.hovering.prev_target = nil
            end
        end
        if G.CONTROLLER.focused and G.CONTROLLER.focused.target and G.CONTROLLER.focused.target.REMOVED then
            G.CONTROLLER.focused.target = nil
        end
        if G.CONTROLLER.clicked and G.CONTROLLER.clicked.target and G.CONTROLLER.clicked.target.REMOVED then
            G.CONTROLLER.clicked.target = nil
        end
        if G.CONTROLLER.dragging and G.CONTROLLER.dragging.target and G.CONTROLLER.dragging.target.REMOVED then
            G.CONTROLLER.dragging.target = nil
        end
    end
end
ShopGen.clear_controller_targets = clear_controller_targets

-- Clear all event queues across all categories
local function clear_all_event_queues()
    if G.E_MANAGER and G.E_MANAGER.queues then
        for _, q in pairs(G.E_MANAGER.queues) do
            if type(q) == 'table' then
                for i = #q, 1, -1 do
                    if not q[i].no_delete then
                        table.remove(q, i)
                    end
                end
            end
        end
    end
end
ShopGen.clear_all_event_queues = clear_all_event_queues

-- Full sweep & GC cycle
function ShopGen.cleanup_cycle()
    prune_engine_tables()
    clear_controller_targets()
    clear_all_event_queues()
    collectgarbage("collect")
    collectgarbage("collect")
end

-- Helper to immediately snap cards and areas to avoid easing/sliding delays
local function snap_card_area(area)
    if not area then return end
    if area.align_cards then area:align_cards() end
    if area.hard_set_VT then area:hard_set_VT() end
    if area.cards then
        for _, card in ipairs(area.cards) do
            if card.hard_set_VT then card:hard_set_VT() end
            if card.children then
                for _, child in pairs(card.children) do
                    if child and child.hard_set_VT then child:hard_set_VT() end
                end
            end
        end
    end
end

-- Populate the top Joker Bar (G.jokers) with 0 to 9 jokers
function G.DATASET_MOD:populate_joker_bar()
    if not G.jokers then return end

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    clear_card_area(G.jokers)

    -- Distribution: 0 to 9 jokers, with 7, 8, 9 appearing less commonly
    -- 0-6: 12% each (84% total)
    -- 7-9: ~5.33% each (16% total)
    local num_jokers
    local r = math.random()
    if r < 0.84 then
        num_jokers = math.random(0, 6)
    else
        num_jokers = math.random(7, 9)
    end

    G.jokers.config.card_limit = math.max(5, num_jokers)

    -- Debuff simulation (e.g. boss blinds Crimson Heart / Verdant Leaf)
    -- 10% chance that 1 or more jokers on the bar are debuffed
    local has_debuffs = (math.random() < 0.10) and (num_jokers > 0)
    local debuff_indices = {}
    if has_debuffs then
        -- 50% chance 1 random joker debuffed, 30% chance 2 debuffed, 20% chance all debuffed (Verdant Leaf)
        local d_roll = math.random()
        if d_roll < 0.50 then
            debuff_indices[math.random(1, num_jokers)] = true
        elseif d_roll < 0.80 then
            local idx1 = math.random(1, num_jokers)
            local idx2 = math.random(1, num_jokers)
            debuff_indices[idx1] = true
            debuff_indices[idx2] = true
        else
            for k = 1, num_jokers do
                debuff_indices[k] = true
            end
        end
    end

    for i = 1, num_jokers do
        local card = self:create_random_joker_card(G.jokers, nil, true)
        if card then
            if debuff_indices[i] then
                if card.set_debuff then
                    card:set_debuff(true)
                else
                    card.debuff = true
                end
            end
            G.jokers:emplace(card)
        end
    end

    if G.jokers.align_cards then
        G.jokers:align_cards()
    end
end

-- =========================================================
-- POPULATE CONSUMABLES AREA (NORMAL & PERKEO MODES)
-- =========================================================
function G.DATASET_MOD:populate_consumables_area()
    if not G.consumeables then return end

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    clear_card_area(G.consumeables)

    local _, tarots, spectrals, planets = self:get_joker_pools()

    -- 10% chance for Perkeo mode
    local is_perkeo = (math.random() < 0.10)
    self.is_perkeo_active = is_perkeo

    -- Slot limit under deck / vouchers (1-3)
    local slot_limit = math.random(1, 3)

    if not is_perkeo then
        -- Normal Circumstances (90% of the time):
        -- 0, 1, or 2 cards
        local num_cards = math.random(0, 2)
        G.consumeables.config.card_limit = slot_limit

        -- 10% chance for Showman effect allowing duplicates
        local has_showman = (math.random() < 0.10)
        local used_centers = {}

        local types = {'Tarot', 'Spectral', 'Planet'}
        for i = 1, num_cards do
            local chosen_type = types[math.random(#types)]
            local pool = (chosen_type == 'Tarot' and tarots) or (chosen_type == 'Spectral' and spectrals) or planets

            local chosen_center = nil
            if pool and #pool > 0 then
                if has_showman then
                    chosen_center = pool[math.random(#pool)]
                else
                    for attempt = 1, 20 do
                        local cand = pool[math.random(#pool)]
                        if not used_centers[cand.key] then
                            chosen_center = cand
                            break
                        end
                    end
                    chosen_center = chosen_center or pool[math.random(#pool)]
                    used_centers[chosen_center.key] = true
                end
            end

            local card = self:create_random_consumable_card(G.consumeables, chosen_type, chosen_center)
            if card then
                -- Sometimes a consumable can be Negative even without full slots (e.g. from Perkeo, Sixth Sense, etc.)
                if math.random() < 0.20 then
                    card:set_edition({negative = true}, true, true)
                end
                G.consumeables:emplace(card)
            end
        end
    else
        -- Perkeo Mode (10% of the time):
        -- Up to configurable max consumables (self.config.perkeo_max_consumables), weighted towards having
        -- the same type appear multiple times, and anything exceeding the slot limit is negative!
        local max_limit = math.max(slot_limit + 1, self.config.perkeo_max_consumables or 10)
        local num_cards = math.random(slot_limit + 1, max_limit)
        G.consumeables.config.card_limit = math.max(slot_limit, num_cards)

        -- Pick the primary duplicated consumable
        local types = {'Tarot', 'Spectral', 'Planet'}
        local dup_type = types[math.random(#types)]
        local dup_pool = (dup_type == 'Tarot' and tarots) or (dup_type == 'Spectral' and spectrals) or planets
        local dup_center = dup_pool and dup_pool[math.random(#dup_pool)]

        for i = 1, num_cards do
            local card_type = dup_type
            local card_center = dup_center

            -- 75% chance primary duplicate, 25% chance another random consumable
            if math.random() > 0.75 or not dup_center then
                card_type = types[math.random(#types)]
                local pool = (card_type == 'Tarot' and tarots) or (card_type == 'Spectral' and spectrals) or planets
                card_center = pool and pool[math.random(#pool)]
            end

            local card = self:create_random_consumable_card(G.consumeables, card_type, card_center)
            if card then
                -- Any card exceeding the consumable slot limit is Negative
                if i > slot_limit then
                    card:set_edition({negative = true}, true, true)
                end
                G.consumeables:emplace(card)
            end
        end
    end

    -- Randomly shuffle the order of consumables in G.consumeables
    if G.consumeables.cards and #G.consumeables.cards > 1 then
        for i = #G.consumeables.cards, 2, -1 do
            local j = math.random(i)
            G.consumeables.cards[i], G.consumeables.cards[j] = G.consumeables.cards[j], G.consumeables.cards[i]
        end
    end

    if G.consumeables.align_cards then
        G.consumeables:align_cards()
    end
end



-- =========================================================
-- SETUP SHOP SCREEN
-- =========================================================
function G.DATASET_MOD:setup_shop_screen()
    if not G.GAME then return end

    if G.booster_pack then
        G.booster_pack:remove()
        G.booster_pack = nil
        G.pack_cards = nil
        prune_engine_tables()
    end

    -- Cycle shop slots: 2 (no voucher) -> 3 (overstock tier 1) -> 4 (overstock tier 2)
    self.shop_slot_cycle = ((self.shop_slot_cycle or 1) % 3) + 2
    local num_slots = self.shop_slot_cycle

    G.GAME.shop = G.GAME.shop or {}
    G.GAME.shop.joker_max = num_slots

    -- Build G.shop UIBox ONCE to prevent leaking G.SHOP_SIGN, AnimatedSprites, and duplicate CardAreas
    if not G.shop then
        if G.UIDEF and G.UIDEF.shop then
            G.shop = UIBox{
                definition = G.UIDEF.shop(),
                config = {align = 'tmi', offset = {x = 0, y = -5.3}, major = G.hand, bond = 'Weak'}
            }
        end
    else
        -- Update slot limits dynamically without destroying and rebuilding G.shop or leaking G.SHOP_SIGN!
        if G.shop_jokers then
            G.shop_jokers.config.card_limit = num_slots
            G.shop_jokers.T.w = num_slots * 1.01 * G.CARD_W
        end
        if G.shop.recalculate then
            G.shop:recalculate()
        end
    end

    -- Unconditionally snap shop offset and state so it is NEVER left offscreen after pack phase
    if G.shop then
        G.shop.alignment.offset.y = -5.3
        if G.shop.hard_set_VT then G.shop:hard_set_VT() end
    end
    G.STATE = G.STATES.SHOP

    if G.shop_jokers then
        clear_card_area(G.shop_jokers)
        G.shop_jokers.config.card_limit = num_slots

        -- 10% chance for Showman effect (allows duplicate cards in shop)
        local has_showman = (math.random() < 0.10)
        local used_shop_keys = {}

        local jokers, tarots, spectrals, planets = self:get_joker_pools()

        -- Populate slots: 70% Joker, 10% Tarot, 10% Spectral, 10% Planet
        -- Note: Legendary jokers do NOT appear in the shop
        local non_leg_jokers = ShopGen.get_all_jokers(false)
        for i = 1, num_slots do
            local roll = math.random()
            local card = nil
            local card_type = 'Joker'

            if roll < 0.70 then
                card_type = 'Joker'
                local chosen_center = nil
                if #non_leg_jokers > 0 then
                    if has_showman then
                        chosen_center = non_leg_jokers[math.random(#non_leg_jokers)]
                    else
                        for attempt = 1, 25 do
                            local cand = non_leg_jokers[math.random(#non_leg_jokers)]
                            if not used_shop_keys[cand.key] then
                                chosen_center = cand; break
                            end
                        end
                        chosen_center = chosen_center or non_leg_jokers[math.random(#non_leg_jokers)]
                        used_shop_keys[chosen_center.key] = true
                    end
                end
                card = self:create_random_joker_card(G.shop_jokers, chosen_center, false)
            else
                local pool
                if roll < 0.80 then
                    card_type = 'Tarot'; pool = tarots
                elseif roll < 0.90 then
                    card_type = 'Spectral'; pool = spectrals
                else
                    card_type = 'Planet'; pool = planets
                end

                local chosen_center = nil
                if pool and #pool > 0 then
                    if has_showman then
                        chosen_center = pool[math.random(#pool)]
                    else
                        for attempt = 1, 25 do
                            local cand = pool[math.random(#pool)]
                            if not used_shop_keys[cand.key] then
                                chosen_center = cand; break
                            end
                        end
                        chosen_center = chosen_center or pool[math.random(#pool)]
                        used_shop_keys[chosen_center.key] = true
                    end
                end
                card = self:create_random_consumable_card(G.shop_jokers, card_type, chosen_center)
            end

            if card then
                attach_shop_card_ui_sync(card)
                G.shop_jokers:emplace(card)
            end
        end

        if G.shop_jokers.align_cards then
            G.shop_jokers:align_cards()
        end
    end

    -- Top Joker Bar (0 to 9 jokers)
    self:populate_joker_bar()

    -- Top Consumable Area (0 to 2 cards, or Perkeo mode)
    self:populate_consumables_area()

    self:randomize_background()

    -- Snap all cards and areas immediately to eliminate sliding animations
    snap_card_area(G.shop_jokers)
    snap_card_area(G.jokers)
    snap_card_area(G.consumeables)
end

-- =========================================================
-- SETUP PACK SCREEN (BUFFOON, ARCANA, SPECTRAL, CELESTIAL)
-- =========================================================
function G.DATASET_MOD:setup_pack_screen()
    if not G.GAME then return end

    if G.shop then
        G.shop.alignment.offset.y = G.ROOM.T.y + 11
        if G.shop.hard_set_VT then G.shop:hard_set_VT() end
    end
    if G.booster_pack then
        G.booster_pack:remove()
        G.booster_pack = nil
        G.pack_cards = nil
        prune_engine_tables()
    end

    -- 80% Buffoon pack, 20% evenly split between Tarot (Arcana), Spectral, and Planet (Celestial)
    local pack_roll = math.random()
    local pack_kind = 'Buffoon'
    if pack_roll >= 0.80 then
        local sub_roll = math.random()
        if sub_roll < 0.333 then
            pack_kind = 'Arcana'
        elseif sub_roll < 0.666 then
            pack_kind = 'Spectral'
        else
            pack_kind = 'Celestial'
        end
    end
    self.current_pack_kind = pack_kind

    -- 10% chance for Showman effect (allows duplicate cards in pack)
    local has_showman = (math.random() < 0.10)
    self.pack_has_showman = has_showman

    local _, base_tarots, base_spectrals, base_planets = self:get_joker_pools()

    -- Legendary jokers NEVER appear in buffoon packs
    local pack_jokers = ShopGen.get_all_jokers(false)

    -- In vanilla:
    -- Arcana packs can spawn 'c_soul' (The Soul) (~0.3% base in vanilla, we give a healthy spawn chance for CV dataset: ~5%)
    -- Spectral packs can spawn any spectral card plus 'c_soul' and 'c_black_hole'
    -- Celestial packs can spawn planet cards plus 'c_black_hole' (Black Hole)
    local soul_center = G.P_CENTERS and G.P_CENTERS['c_soul']
    local black_hole_center = G.P_CENTERS and G.P_CENTERS['c_black_hole']

    local pack_size = 2
    local uibox_fn = nil
    local pack_state = G.STATES.BUFFOON_PACK
    local card_type = 'Joker'
    local pool = pack_jokers

    if pack_kind == 'Buffoon' then
        -- Buffoon Pack: 2 (Normal), 4 (Jumbo/Mega)
        self.pack_size_cycle = (self.pack_size_cycle == 2) and 4 or 2
        pack_size = self.pack_size_cycle
        uibox_fn = create_UIBox_buffoon_pack
        pack_state = G.STATES.BUFFOON_PACK
        card_type = 'Joker'
        pool = pack_jokers
    elseif pack_kind == 'Arcana' then
        -- Arcana Pack: 3 (Normal), 5 (Jumbo/Mega)
        self.arcana_size_cycle = (self.arcana_size_cycle == 3) and 5 or 3
        pack_size = self.arcana_size_cycle
        uibox_fn = create_UIBox_arcana_pack
        pack_state = G.STATES.TAROT_PACK
        card_type = 'Tarot'
        pool = {}
        for _, c in ipairs(base_tarots) do table.insert(pool, c) end
        if soul_center then table.insert(pool, soul_center) end
    elseif pack_kind == 'Spectral' then
        -- Spectral Pack: 2 (Normal), 4 (Jumbo/Mega)
        self.spectral_size_cycle = (self.spectral_size_cycle == 2) and 4 or 2
        pack_size = self.spectral_size_cycle
        uibox_fn = create_UIBox_spectral_pack
        pack_state = G.STATES.SPECTRAL_PACK
        card_type = 'Spectral'
        pool = {}
        for _, c in ipairs(base_spectrals) do table.insert(pool, c) end
        if soul_center then table.insert(pool, soul_center) end
        if black_hole_center then table.insert(pool, black_hole_center) end
    elseif pack_kind == 'Celestial' then
        -- Celestial Pack: 3 (Normal), 5 (Jumbo/Mega)
        self.celestial_size_cycle = (self.celestial_size_cycle == 3) and 5 or 3
        pack_size = self.celestial_size_cycle
        uibox_fn = create_UIBox_celestial_pack
        pack_state = G.STATES.PLANET_PACK
        card_type = 'Planet'
        pool = {}
        for _, c in ipairs(base_planets) do table.insert(pool, c) end
        if black_hole_center then table.insert(pool, black_hole_center) end
    end

    G.GAME.pack_size = pack_size
    G.GAME.pack_choices = (pack_size >= 4 and 2) or 1
    self.current_pack_size = pack_size

    if uibox_fn then
        G.booster_pack = UIBox{
            definition = uibox_fn(),
            config = {align = "tmi", offset = {x = 0, y = -2.2}, major = G.hand, bond = 'Weak'}
        }
        G.booster_pack.alignment.offset.y = -2.2
        if G.booster_pack.hard_set_VT then G.booster_pack:hard_set_VT() end
        G.STATE = pack_state
    end

    if G.pack_cards then
        clear_card_area(G.pack_cards)
        G.pack_cards.config.card_limit = pack_size

        local used_pack_keys = {}

        for i = 1, pack_size do
            local chosen_center = nil
            if pool and #pool > 0 then
                if has_showman then
                    chosen_center = pool[math.random(#pool)]
                else
                    for attempt = 1, 30 do
                        local cand = pool[math.random(#pool)]
                        if not used_pack_keys[cand.key] then
                            chosen_center = cand; break
                        end
                    end
                    chosen_center = chosen_center or pool[math.random(#pool)]
                    used_pack_keys[chosen_center.key] = true
                end
            end

            local card
            if card_type == 'Joker' then
                card = self:create_random_joker_card(G.pack_cards, chosen_center, false)
            else
                local set_name = (chosen_center and chosen_center.set) or card_type
                card = self:create_random_consumable_card(G.pack_cards, set_name, chosen_center)
            end

            if card then
                G.pack_cards:emplace(card)
            end
        end

        if G.pack_cards.align_cards then
            G.pack_cards:align_cards()
        end
    end

    -- Top Joker Bar (0 to 9 jokers)
    self:populate_joker_bar()

    -- Top Consumable Area (0 to 2 cards, or Perkeo mode)
    self:populate_consumables_area()

    self:randomize_background()

    -- Snap all cards and areas immediately to eliminate sliding animations
    snap_card_area(G.pack_cards)
    snap_card_area(G.jokers)
    snap_card_area(G.consumeables)
end

-- Step alternating generator: 100 on shop screen, then 100 on pack screen
function G.DATASET_MOD:generate_shop_step()
    self.shop_phase = self.shop_phase or 'shop'
    self.phase_step = self.phase_step or 0

    if self.shop_phase == 'shop' then
        self:setup_shop_screen()
        self.phase_step = self.phase_step + 1
        if self.phase_step >= (self.config.frames_per_phase or 100) then
            self.shop_phase = 'pack'
            self.phase_step = 0
            print("DATASET: Completed 100 shop screens, switching to PACK phase")
        end
    else
        self:setup_pack_screen()
        self.phase_step = self.phase_step + 1
        if self.phase_step >= (self.config.frames_per_phase or 100) then
            self.shop_phase = 'shop'
            self.phase_step = 0
            print("DATASET: Completed 100 pack screens, switching to SHOP phase")
        end
    end
    ShopGen.cleanup_cycle()
end

-- Toggle screen between Shop and Pack for manual testing/calibration (press 'P')
function G.DATASET_MOD:toggle_shop_screen_mode()
    if self.shop_phase == 'shop' then
        self.shop_phase = 'pack'
        self:setup_pack_screen()
    else
        self.shop_phase = 'shop'
        self:setup_shop_screen()
    end
    self.capture_flash = 1.0
    self.capture_flash_text = "SWITCHED TO " .. string.upper(self.shop_phase) .. " VIEW (" .. (self.current_pack_kind or 'SHOP') .. ")"
end

return ShopGen
