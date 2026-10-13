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
