-- gamestate_dataset_cv/src/ui.lua
-- Contains SMODS settings page (config_tab), in-game calibration overlay, and keypressed handlers.

local UI = {}

function G.DATASET_MOD:get_selected_object()
    local t = self.selected_target or 'roi_joker_bar'
    if t == 'roi_joker_bar' then return self.config.rois.joker_bar, 'roi', "ROI: Joker Bar"
    elseif t == 'roi_shop' then return self.config.rois.shop, 'roi', "ROI: Shop Area"
    elseif t == 'roi_pack' then return self.config.rois.pack, 'roi', "ROI: Pack Area"
    elseif t == 'roi_consumeables' then return self.config.rois.consumeables, 'roi', "ROI: Consumables (Top Right)"
    elseif t == 'roi_card_hand' then return self.config.rois.card_hand, 'roi', "ROI: Card Hand"
    elseif t == 'box_joker_bar' then return self.config.boxes.joker_bar_card, 'box', "Box: Joker Bar Card"
    elseif t == 'box_item' then return self.config.boxes.item_card, 'box', "Box: Shop/Pack Item"
    elseif t == 'box_consumeable' then return self.config.boxes.consumeable_card, 'box', "Box: Consumable Card"
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
end

function UI.create_config_tab()
    local mod = G.DATASET_MOD
    local current_mode_idx = (mod.config.mode == 'card_classifier' and 2) or (mod.config.mode == 'shop_classifier' and 3) or 1
    local current_calib_idx = (mod.config.calibration_mode == 'generation' and 2) or 1

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
            minw = 8.5,
            align = "cm",
            padding = 0.2,
            colour = G.C.BLACK
        },
        nodes = {
            -- Title
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {n = G.UIT.T, config = {text = "GameState Dataset CV - Settings", scale = 0.6, colour = G.C.GOLD}}
                }
            },
            -- Mode Selector Cycle
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {n = G.UIT.T, config = {text = "Classifier Mode: ", scale = 0.45, colour = G.C.UI.TEXT_LIGHT}},
                    create_option_cycle({
                        id = 'gdcv_mode_cycle',
                        options = {'Off', 'Card Classifier', 'Shop Classifier'},
                        current_option = current_mode_idx,
                        opt_callback = 'gdcv_change_mode',
                        w = 4.5,
                        scale = 0.8
                    })
                }
            },
            -- Calibration / Generation Cycle
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {n = G.UIT.T, config = {text = "Operation State: ", scale = 0.45, colour = G.C.UI.TEXT_LIGHT}},
                    create_option_cycle({
                        id = 'gdcv_state_cycle',
                        options = {'Setup Mode', 'Generation Mode'},
                        current_option = current_calib_idx,
                        opt_callback = 'gdcv_change_calib_state',
                        w = 4.5,
                        scale = 0.8
                    })
                }
            },
            -- Target Frames Cap Cycle
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {n = G.UIT.T, config = {text = "Target Frame Cap: ", scale = 0.45, colour = G.C.UI.TEXT_LIGHT}},
                    create_option_cycle({
                        id = 'gdcv_mf_cycle',
                        options = max_frames_options,
                        current_option = current_mf_idx,
                        opt_callback = 'gdcv_change_max_frames',
                        w = 4.5,
                        scale = 0.8
                    })
                }
            },
            -- Perkeo Max Consumables Cycle
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {n = G.UIT.T, config = {text = "Perkeo Max Consumables: ", scale = 0.45, colour = G.C.UI.TEXT_LIGHT}},
                    create_option_cycle({
                        id = 'gdcv_pm_cycle',
                        options = perkeo_max_options,
                        current_option = current_pm_idx,
                        opt_callback = 'gdcv_change_perkeo_max',
                        w = 4.5,
                        scale = 0.8
                    })
                }
            },
            -- Stats & Info Row
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.1},
                nodes = {
                    {
                        n = G.UIT.T,
                        config = {
                            text = string.format("Card Frames: %d  |  Shop Frames: %d  |  Phase: %s",
                                mod.frame_counts.card or 0,
                                mod.frame_counts.shop or 0,
                                string.upper(mod.shop_phase or 'shop')),
                            scale = 0.38,
                            colour = G.C.UI.TEXT_INACTIVE
                        }
                    }
                }
            },
            -- Buttons Row: Save & Load Config
            {
                n = G.UIT.R,
                config = {align = "cm", padding = 0.15},
                nodes = {
                    UIBox_button({
                        button = 'gdcv_save_config',
                        label = {'Save Config'},
                        minw = 2.4,
                        minh = 0.7,
                        scale = 0.38,
                        colour = G.C.GREEN
                    }),
                    {n = G.UIT.B, config = {w = 0.3, h = 0.1}},
                    UIBox_button({
                        button = 'gdcv_load_config',
                        label = {'Load Config'},
                        minw = 2.4,
                        minh = 0.7,
                        scale = 0.38,
                        colour = G.C.ORANGE
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

        if (self.shop_phase or 'shop') == 'shop' then
            draw_roi_box(self.config.rois.shop, "[ROI: SHOP SLOTS (UPPER RIGHT)]", sel == 'roi_shop', 1, 0.8, 0.2)
        else
            local p_name = string.format("[ROI: %s PACK]", string.upper(self.current_pack_kind or 'PACK'))
            draw_roi_box(self.config.rois.pack, p_name, sel == 'roi_pack', 1, 0.3, 0.8)
        end
    end

    -- 2. Draw Card Bounding Boxes Polygons
    if self.captured_boxes then
        for card, box in pairs(self.captured_boxes) do
            if box and box.points then
                local pts = box.points
                local is_sel_box = (self.selected_target == 'box_joker_bar' and box.box_name == 'joker_bar_card') or
                                   (self.selected_target == 'box_item' and box.box_name == 'item_card') or
                                   (self.selected_target == 'box_consumeable' and box.box_name == 'consumeable_card') or
                                   (self.selected_target == 'box_card' and box.box_name == 'card')

                if box.box_name == 'joker_bar_card' then
                    love.graphics.setColor(0, 1, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'consumeable_card' then
                    love.graphics.setColor(1, 0.4, 1, is_sel_box and 1.0 or 0.6)
                elseif box.box_name == 'item_card' then
                    love.graphics.setColor(1, 1, 0, is_sel_box and 1.0 or 0.6)
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
        love.graphics.print("Keys: 1: Joker Bar | 2: Shop | 3: Pack | 4: Consumables | 5: Box Joker | 6: Box Item | 7: Box Consumable", 18, 58)
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
            elseif key == '3' then self.selected_target = 'roi_pack'
            elseif key == '4' then self.selected_target = 'roi_consumeables'
            elseif key == '5' then self.selected_target = 'box_joker_bar'
            elseif key == '6' then self.selected_target = 'box_item'
            elseif key == '7' then self.selected_target = 'box_consumeable'
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
    if self.config.mode == 'off' then return end

    if self.capture_flash and self.capture_flash > 0 then
        self.capture_flash = math.max(0, self.capture_flash - dt)
    end

    if self.config.calibration_mode == 'setup' then
        local obj, obj_type = self:get_selected_object()
        if not obj then return end

        if obj_type == 'roi' then
            local roi_step = love.keyboard.isDown('lshift', 'rshift') and (500 * dt) or (150 * dt)
            if love.keyboard.isDown('w') then obj.y = obj.y - roi_step end
            if love.keyboard.isDown('s') then obj.y = obj.y + roi_step end
            if love.keyboard.isDown('a') then obj.x = obj.x - roi_step end
            if love.keyboard.isDown('d') then obj.x = obj.x + roi_step end
            if love.keyboard.isDown('up') then obj.h = obj.h - roi_step end
            if love.keyboard.isDown('down') then obj.h = obj.h + roi_step end
            if love.keyboard.isDown('left') then obj.w = obj.w - roi_step end
            if love.keyboard.isDown('right') then obj.w = obj.w + roi_step end
        elseif obj_type == 'box' then
            local step = love.keyboard.isDown('lshift', 'rshift') and (0.5 * dt) or (0.1 * dt)
            if love.keyboard.isDown('w') then obj.oy = obj.oy - step end
            if love.keyboard.isDown('s') then obj.oy = obj.oy + step end
            if love.keyboard.isDown('a') then obj.ox = obj.ox - step end
            if love.keyboard.isDown('d') then obj.ox = obj.ox + step end
            if love.keyboard.isDown('up') then obj.h = obj.h - step end
            if love.keyboard.isDown('down') then obj.h = obj.h + step end
            if love.keyboard.isDown('left') then obj.w = obj.w - step end
            if love.keyboard.isDown('right') then obj.w = obj.w + step end
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
            self.timer = 0.25 -- Wait for animations/lerp to settle
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
