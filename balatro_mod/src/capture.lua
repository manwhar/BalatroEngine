-- gamestate_dataset_cv/src/capture.lua
-- Handles coordinate transformations, bounding box calculations, screenshot cropping, and dataset label export.

local Capture = {}

local function apply_local_transform(lx, ly, x, y, r, sx, sy, ox, oy, kx, ky)
    x = x or 0; y = y or 0; r = r or 0; sx = sx or 1; sy = sy or 1
    ox = ox or 0; oy = oy or 0; kx = kx or 0; ky = ky or 0
    lx = lx - ox
    ly = ly - oy
    local slx = lx + ky * ly
    local sly = kx * lx + ly
    lx, ly = slx, sly
    lx = lx * sx
    ly = ly * sy
    local c = math.cos(r)
    local s = math.sin(r)
    local rlx = lx * c - ly * s
    local rly = lx * s + ly * c
    lx, ly = rlx, rly
    lx = lx + x
    ly = ly + y
    return lx, ly
end

local function clamp_aabb_to_roi(roi, bx, by, bw, bh)
    local min_x = math.max(bx, roi.x)
    local max_x = math.min(bx + bw, roi.x + roi.w)
    local min_y = math.max(by, roi.y)
    local max_y = math.min(by + bh, roi.y + roi.h)
    
    if min_x >= max_x or min_y >= max_y then return nil end
    return min_x, min_y, max_x - min_x, max_y - min_y
end

local function to_yolo(roi, bx, by, bw, bh)
    local cx = bx + bw / 2
    local cy = by + bh / 2
    local nx = (cx - roi.x) / roi.w
    local ny = (cy - roi.y) / roi.h
    local nw = bw / roi.w
    local nh = bh / roi.h
    return nx, ny, nw, nh
end

local function json_escape(str)
    if not str then return "" end
    str = tostring(str)
    str = string.gsub(str, "\\", "\\\\")
    str = string.gsub(str, '"', '\\"')
    str = string.gsub(str, "\n", "\\n")
    str = string.gsub(str, "\r", "\\r")
    return str
end

function G.DATASET_MOD:calculate_boxes_for_card(card, draw_args)
    local ox, oy = draw_args[6], draw_args[7]
    local cw = (ox and ox > 0) and (ox * 2) or 71
    local ch = (oy and oy > 0) and (oy * 2) or 95

    -- Select the appropriate bounding box configuration
    local box_def = nil
    local box_name = "card"

    if card.area == G.jokers then
        box_def = self.config.boxes.joker_bar_card
        box_name = "joker_bar_card"
    elseif card.area == G.consumeables then
        -- In Perkeo mode with > 2 cards, use left ~half box to prevent overlapping AABB collision
        if G.consumeables and G.consumeables.cards and #G.consumeables.cards > 2 then
            box_def = {w = 0.55, h = 1.0, ox = -0.225, oy = 0.0}
        else
            box_def = self.config.boxes.consumeable_card or {w = 1.0, h = 1.0, ox = 0.0, oy = 0.0}
        end
        box_name = "consumeable_card"
    elseif card.area == G.shop_jokers or card.area == G.pack_cards then
        box_def = self.config.boxes.item_card
        box_name = "item_card"
    else
        box_def = self.config.boxes.card
        box_name = "card"
    end

    box_def = box_def or {w = 1.0, h = 1.0, ox = 0.0, oy = 0.0}

    local center_x = cw / 2 + box_def.ox * cw
    local center_y = ch / 2 + box_def.oy * ch
    local hw = (box_def.w * cw) / 2
    local hh = (box_def.h * ch) / 2

    local tex_pts = {
        {center_x - hw, center_y - hh},
        {center_x + hw, center_y - hh},
        {center_x + hw, center_y + hh},
        {center_x - hw, center_y + hh}
    }

    local transformed_pts = {}
    for _, pt in ipairs(tex_pts) do
        local lx, ly = apply_local_transform(pt[1], pt[2], unpack(draw_args))
        local gx, gy = love.graphics.transformPoint(lx, ly)
        table.insert(transformed_pts, gx)
        table.insert(transformed_pts, gy)
    end

    local aabb = {
        math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
        math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]),
        math.max(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]) - math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
        math.max(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]) - math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8])
    }

    self.captured_boxes = self.captured_boxes or setmetatable({}, {__mode = "k"})
    self.captured_boxes[card] = {
        box_name = box_name,
        points = transformed_pts,
        aabb = aabb
    }
end

-- ==========================================
-- CARD CLASSIFIER CAPTURE & EXPORT
-- ==========================================
function G.DATASET_MOD:capture_frame_card()
    local roi = self.config.rois.card_hand or {x = 508, y = 567, w = 1071, h = 322}
    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)
    local current_frame = self.frame_counts.card

    love.graphics.captureScreenshot(function(img)
        local cropped = love.image.newImageData(rw, rh)
        cropped:paste(img, 0, 0, rx, ry, rw, rh)
        
        local filedata = cropped:encode("png")
        local raw_bytes = filedata:getString()
        
        local filename = string.format("Mods/gamestate_dataset_cv/dataset/images/frame_%04d.png", current_frame)
        love.filesystem.write(filename, raw_bytes)
        
        if filedata and filedata.release then filedata:release() end
        if cropped and cropped.release then cropped:release() end
        if img and img.release then img:release() end
    end)

    self:save_labels_card(current_frame)
    self.frame_counts.card = self.frame_counts.card + 1
    self.frame_count = self.frame_counts.card
    self.state = 'idle'
    self.capture_flash = 1.0
    self.capture_flash_text = string.format("CARD FRAME %04d SAVED!", current_frame)

    if self.frame_count % 25 == 0 then
        collectgarbage("collect")
    end
end

function G.DATASET_MOD:save_labels_card(frame_num)
    local roi = self.config.rois.card_hand or {x = 508, y = 567, w = 1071, h = 322}
    local yolo_labels = {}
    local json_cards = {}

    if G.hand and G.hand.cards then
        for idx, card in ipairs(G.hand.cards) do
            local box_data = self.captured_boxes and self.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])
                    
                    -- Single class 0 ("card")
                    table.insert(yolo_labels, string.format("0 %.6f %.6f %.6f %.6f", nx, ny, nw, nh))
                    
                    local is_face_down = (card.facing == 'back' or card.sprite_facing == 'back')
                    local rank = "None"
                    local suit = "None"
                    local edition = "None"
                    local enhancement = "None"
                    local seal = "None"
                    local debuffed = false

                    if not is_face_down then
                        if card.config.center and card.config.center.name == "Stone Card" then
                            rank = "Stone"
                            suit = "None"
                        elseif card.base then
                            rank = tostring(card.base.value or "None")
                            suit = tostring(card.base.suit or "None")
                        end
                        
                        if card.edition then
                            if card.edition.foil then edition = "Foil"
                            elseif card.edition.holo then edition = "Holo"
                            elseif card.edition.polychrome then edition = "Polychrome"
                            elseif card.edition.negative then edition = "Negative"
                            end
                        end
                        
                        if card.config.center and card.config.center.name ~= "Default Base" and card.config.center.name ~= "Stone Card" then
                            enhancement = string.gsub(card.config.center.name, " Card", "")
                            enhancement = string.gsub(enhancement, " ", "")
                        elseif card.config.center and card.config.center.name == "Stone Card" then
                            enhancement = "Stone"
                        end
                        
                        seal = card.seal or "None"
                        debuffed = (card.debuff == true)
                    end
                    
                    local selected = card.highlighted and true or false
                    local card_skin = "None"
                    local is_suit_hc = false
                    if suit ~= "None" and not is_face_down then
                        card_skin = (G.SETTINGS.CUSTOM_DECK and G.SETTINGS.CUSTOM_DECK.Collabs and G.SETTINGS.CUSTOM_DECK.Collabs[suit]) or "default"
                        is_suit_hc = (G.SETTINGS.colour_palettes and G.SETTINGS.colour_palettes[suit] == "hc") or false
                    end
                    
                    table.insert(json_cards, string.format(
                        '  {"card_index": %d, "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f], "rank": "%s", "suit": "%s", "edition": "%s", "enhancement": "%s", "seal": "%s", "selected": %s, "debuffed": %s, "face_down": %s, "skin": "%s", "high_contrast": %s}',
                        idx, nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4],
                        rank, suit, edition, enhancement, seal, selected and "true" or "false", debuffed and "true" or "false", is_face_down and "true" or "false",
                        json_escape(card_skin), is_suit_hc and "true" or "false"
                    ))
                end
            end
        end
    end

    local label_path = string.format("Mods/gamestate_dataset_cv/dataset/labels/frame_%04d.txt", frame_num)
    love.filesystem.write(label_path, table.concat(yolo_labels, "\n"))

    local json_path = string.format("Mods/gamestate_dataset_cv/dataset/labels/frame_%04d.json", frame_num)
    local json_content = "[\n" .. table.concat(json_cards, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)
end

-- ==========================================
-- SHOP, PACK & CONSUMABLE EXPORT
-- ==========================================
local function get_rarity_str(card)
    local r = card.config and card.config.center and card.config.center.rarity
    if r == 1 then return "Common"
    elseif r == 2 then return "Uncommon"
    elseif r == 3 then return "Rare"
    elseif r == 4 then return "Legendary"
    end
    return tostring(r or "Common")
end

local function get_edition_str(card)
    if not card.edition then return "None" end
    if card.edition.foil then return "Foil"
    elseif card.edition.holo then return "Holo"
    elseif card.edition.polychrome then return "Polychrome"
    elseif card.edition.negative then return "Negative"
    end
    return "None"
end

local function export_roi_dataset(mod, img, roi, card_list, area_name, current_frame, file_suffix)
    if not roi then return end
    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)

    -- 1. Save Image Crop
    local cropped = love.image.newImageData(rw, rh)
    cropped:paste(img, 0, 0, rx, ry, rw, rh)
    local filedata = cropped:encode("png")
    local img_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/images/frame_%04d_%s.png", current_frame, file_suffix)
    love.filesystem.write(img_path, filedata:getString())

    if filedata and filedata.release then filedata:release() end
    if cropped and cropped.release then cropped:release() end

    -- 2. Build YOLO and JSON labels
    local yolo_labels = {}
    local json_entries = {}

    local class_id_map = {
        ['Joker'] = 0,
        ['Tarot'] = 1,
        ['Spectral'] = 2,
        ['Planet'] = 3
    }

    if card_list then
        for idx, card in ipairs(card_list) do
            local box_data = mod.captured_boxes and mod.captured_boxes[card]
            if box_data and box_data.aabb then
                local min_x, min_y, bw, bh = box_data.aabb[1], box_data.aabb[2], box_data.aabb[3], box_data.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])

                    local card_type = (card.ability and card.ability.set) or (card.config and card.config.center and card.config.center.set) or "Joker"
                    local class_id = class_id_map[card_type] or 0

                    table.insert(yolo_labels, string.format("%d %.6f %.6f %.6f %.6f", class_id, nx, ny, nw, nh))

                    local name = (card.config and card.config.center and card.config.center.name) or "Unknown"
                    local key = (card.config and card.config.center and card.config.center.key) or "unknown"
                    local rarity = get_rarity_str(card)
                    local edition = get_edition_str(card)
                    local is_negative = (card.edition and card.edition.negative == true) or (card.ability and card.ability.negative == true)
                    local is_eternal = (card.ability and card.ability.eternal == true)
                    local is_perishable = (card.ability and card.ability.perishable == true)
                    local is_rental = (card.ability and card.ability.rental == true)
                    local is_debuffed = (card.debuff == true)
                    local art_variant = card.custom_art_variant or "default"

                    table.insert(json_entries, string.format(
                        '  {"card_index": %d, "area": "%s", "card_type": "%s", "name": "%s", "key": "%s", "rarity": "%s", "edition": "%s", "negative": %s, "eternal": %s, "perishable": %s, "rental": %s, "debuffed": %s, "art_variant": "%s", "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f]}',
                        idx, area_name, card_type, json_escape(name), json_escape(key), rarity, edition,
                        is_negative and "true" or "false", is_eternal and "true" or "false", is_perishable and "true" or "false", is_rental and "true" or "false", is_debuffed and "true" or "false",
                        json_escape(art_variant),
                        nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4]
                    ))
                end
            end
        end
    end

    -- Write YOLO .txt label
    local txt_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/labels/frame_%04d_%s.txt", current_frame, file_suffix)
    love.filesystem.write(txt_path, table.concat(yolo_labels, "\n"))

    -- Write JSON label
    local json_path = string.format("Mods/gamestate_dataset_cv/dataset_shop/labels/frame_%04d_%s.json", current_frame, file_suffix)
    local json_content = "[\n" .. table.concat(json_entries, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)
end

function G.DATASET_MOD:capture_frame_shop()
    local current_frame = self.frame_counts.shop
    local current_phase = self.shop_phase or 'shop'

    love.graphics.captureScreenshot(function(img)
        -- 1. Always capture top Joker Bar
        local roi_joker = self.config.rois.joker_bar
        local joker_cards = (G.jokers and G.jokers.cards) or {}
        export_roi_dataset(self, img, roi_joker, joker_cards, "joker_bar", current_frame, "joker")

        -- 2. Always capture top Consumable Area
        local roi_cons = self.config.rois.consumeables
        local cons_cards = (G.consumeables and G.consumeables.cards) or {}
        export_roi_dataset(self, img, roi_cons, cons_cards, "consumeables", current_frame, "consumeable")

        -- 3. Capture active screen ROI (Shop or Pack)
        if current_phase == 'shop' then
            local roi_shop = self.config.rois.shop
            local shop_cards = (G.shop_jokers and G.shop_jokers.cards) or {}
            export_roi_dataset(self, img, roi_shop, shop_cards, "shop", current_frame, "shop")
        else
            local roi_pack = self.config.rois.pack
            local pack_cards = (G.pack_cards and G.pack_cards.cards) or {}
            export_roi_dataset(self, img, roi_pack, pack_cards, "pack", current_frame, "pack")
        end

        if img and img.release then img:release() end
    end)

    self.frame_counts.shop = self.frame_counts.shop + 1
    self.frame_count = self.frame_counts.shop
    self.state = 'idle'
    self.capture_flash = 1.0
    self.capture_flash_text = string.format("SHOP FRAME %04d SAVED (%s)!", current_frame, string.upper(current_phase))

    if self.frame_count % 25 == 0 then
        collectgarbage("collect")
    end
end

-- Top-level capture dispatcher
function G.DATASET_MOD:capture_frame()
    if self.config.mode == 'card_classifier' then
        self:capture_frame_card()
    elseif self.config.mode == 'shop_classifier' then
        self:capture_frame_shop()
    end
end

return Capture
