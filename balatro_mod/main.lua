-- gamestate_dataset_cv/main.lua

G.DATASET_MOD = {
    config = {
        roi = {x = 300, y = 200, w = 1320, h = 680}, -- Default screen-space ROI (adjustable)
        boxes = {
            card = {w = 1.0, h = 1.0, ox = 0, oy = 0}
        },
        mode = 'setup' -- 'setup' or 'generation'
    },
    is_generating = false,
    frame_count = 0,
    classes = {},
    class_mapping = {},
    next_class_id = 0,
    captured_transforms = {},
    current_card = nil,
    state = 'idle',
    timer = 0,
    mod_path = "",
    selected_box = "roi" -- for setup UI
}

local function get_mod_path()
    local p = love.filesystem.getSaveDirectory() .. "/Mods/gamestate_dataset_cv/"
    if SMODS.current_mod and SMODS.current_mod.path then
        local smods_path = SMODS.current_mod.path
        if string.sub(smods_path, 2, 2) == ":" or string.sub(smods_path, 1, 1) == "/" then
            p = smods_path
        end
    end
    if string.sub(p, -1) ~= "/" and string.sub(p, -1) ~= "\\" then
        p = p .. "/"
    end
    return p
end
G.DATASET_MOD.mod_path = get_mod_path()

function G.DATASET_MOD:save_config()
    local content = "return {\n"
    content = content .. string.format("  roi = {x=%f, y=%f, w=%f, h=%f},\n", self.config.roi.x, self.config.roi.y, self.config.roi.w, self.config.roi.h)
    content = content .. "  boxes = {\n"
    for k,v in pairs(self.config.boxes) do
        content = content .. string.format("    %s = {w=%f, h=%f, ox=%f, oy=%f},\n", k, v.w, v.h, v.ox, v.oy)
    end
    content = content .. "  }\n}\n"
    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset")
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/boxes.lua", content)
    print("DATASET: Config saved!")
end

function G.DATASET_MOD:load_config()
    local path = "Mods/gamestate_dataset_cv/dataset/boxes.lua"
    if love.filesystem.getInfo(path) then
        local content = love.filesystem.read(path)
        local chunk = loadstring(content)
        if chunk then
            local t = chunk()
            if t and t.roi and t.boxes then
                self.config.roi = t.roi
                self.config.boxes = t.boxes
                print("DATASET: Config loaded!")
            end
        end
    end
end

-- Class Management
function G.DATASET_MOD:load_classes()
    local path = "Mods/gamestate_dataset_cv/dataset/classes.txt"
    if love.filesystem.getInfo(path) then
        local content = love.filesystem.read(path)
        if content then
            local id = 0
            for line in content:gmatch("[^\r\n]+") do
                local trimmed = line:match("^%s*(.-)%s*$")
                if trimmed and trimmed ~= "" then
                    self.class_mapping[trimmed] = id
                    id = id + 1
                end
            end
            self.next_class_id = id
            print("DATASET: Loaded " .. id .. " classes from classes.txt")
        end
    end
end

function G.DATASET_MOD:init_frame_count()
    local max_idx = -1
    local dirs = {"Mods/gamestate_dataset_cv/dataset/images", "Mods/gamestate_dataset_cv/dataset/labels"}
    for _, dir in ipairs(dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local num = f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_idx then
                        max_idx = n
                    end
                end
            end
        end
    end
    self.frame_count = max_idx + 1
    print("DATASET: Resuming frame_count from " .. self.frame_count)
end

-- Try loading config on start
G.DATASET_MOD:load_config()

-- Ensure directories exist via LÖVE to bypass OS path constraints
love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset")
love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset/images")
love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset/labels")

-- Load classes and resume frame counter if restarting
G.DATASET_MOD:load_classes()
G.DATASET_MOD:init_frame_count()

-- Utility for Normal Distribution
function G.DATASET_MOD:normal_dist(mean, std_dev)
    local u1 = math.random()
    local u2 = math.random()
    local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0 * std_dev + mean
end

function G.DATASET_MOD:get_class_id(class_name)
    if not self.class_mapping[class_name] then
        self.class_mapping[class_name] = self.next_class_id
        self.next_class_id = self.next_class_id + 1
        self:save_classes()
    end
    return self.class_mapping[class_name]
end

function G.DATASET_MOD:save_classes()
    local ordered = {}
    for name, id in pairs(self.class_mapping) do
        ordered[id] = name
    end
    local lines = {}
    for i=0, self.next_class_id-1 do
        table.insert(lines, ordered[i])
    end
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/classes.txt", table.concat(lines, "\n"))
end

-- Randomization Helpers
function G.DATASET_MOD:get_random_card_center()
    local cards = {}
    for k, v in pairs(G.P_CARDS) do
        table.insert(cards, v)
    end
    if #cards > 0 then
        return cards[math.random(#cards)]
    end
    return nil
end

function G.DATASET_MOD:get_random_edition()
    local r = math.random()
    if r < 0.2 then return 'foil'
    elseif r < 0.4 then return 'holo'
    elseif r < 0.6 then return 'polychrome'
    end
    return nil
end

function G.DATASET_MOD:get_random_enhancement()
    if math.random() < 0.5 then return nil end
    local enhs = {'m_bonus', 'm_mult', 'm_wild', 'm_glass', 'm_steel', 'm_gold', 'm_lucky'}
    return enhs[math.random(#enhs)]
end

function G.DATASET_MOD:get_random_seal()
    if math.random() < 0.5 then return nil end
    local seals = {'Gold', 'Red', 'Blue', 'Purple'}
    return seals[math.random(#seals)]
end

function G.DATASET_MOD:randomize_background()
    if G.C and G.C.BACKGROUND then
        local realistic_bgs = {
            -- Green (Default / Main Menu)
            {L = {0.15, 0.35, 0.25, 1}, D = {0.08, 0.25, 0.15, 1}},
            -- Blue (Small Blind)
            {L = {0.15, 0.45, 0.75, 1}, D = {0.05, 0.25, 0.45, 1}},
            -- Orange/Yellow (Big Blind)
            {L = {0.85, 0.55, 0.15, 1}, D = {0.55, 0.25, 0.05, 1}},
            -- Red (Boss Blind)
            {L = {0.75, 0.25, 0.25, 1}, D = {0.45, 0.05, 0.05, 1}},
            -- Purple (Tarot / Magic)
            {L = {0.55, 0.25, 0.65, 1}, D = {0.25, 0.05, 0.35, 1}},
            -- Teal (Planet)
            {L = {0.15, 0.65, 0.55, 1}, D = {0.05, 0.35, 0.25, 1}},
            -- Ghostly Blue (Spectral)
            {L = {0.35, 0.65, 0.75, 1}, D = {0.15, 0.35, 0.45, 1}},
            -- Dark / Boss Special
            {L = {0.2, 0.2, 0.2, 1}, D = {0.05, 0.05, 0.05, 1}}
        }
        
        local bg = realistic_bgs[math.random(#realistic_bgs)]
        -- Add slight noise for data augmentation, but keep it looking like the vanilla bg
        local noise = (math.random() - 0.5) * 0.05
        
        G.C.BACKGROUND.L = {math.max(0, math.min(1, bg.L[1] + noise)), math.max(0, math.min(1, bg.L[2] + noise)), math.max(0, math.min(1, bg.L[3] + noise)), 1}
        G.C.BACKGROUND.D = {math.max(0, math.min(1, bg.D[1] + noise)), math.max(0, math.min(1, bg.D[2] + noise)), math.max(0, math.min(1, bg.D[3] + noise)), 1}
        G.C.BACKGROUND.C = G.C.BACKGROUND.L
    end
end

function G.DATASET_MOD:generate_random_hand()
    if not G.hand then return end
    
    -- Reset captured boxes to release all old Card references
    self.captured_boxes = setmetatable({}, {__mode = "k"})
    
    -- Clear current hand
    for i = #G.hand.cards, 1, -1 do
        local c = G.hand.cards[i]
        c:remove()
    end
    G.hand.cards = {}
    G.hand.highlighted = {}
    
    local num_cards = math.max(1, math.min(15, math.floor(self:normal_dist(8, 2))))
    G.hand.config.card_limit = num_cards
    
    for i=1, num_cards do
        local base_card = self:get_random_card_center()
        local center = G.P_CENTERS.c_base
        
        if math.random() < 0.1 and G.P_CENTERS['m_stone'] then
            center = G.P_CENTERS['m_stone']
        else
            local enhancement = self:get_random_enhancement()
            if enhancement and G.P_CENTERS[enhancement] then
                center = G.P_CENTERS[enhancement]
            end
        end
        
        local card = Card(G.play.T.x + G.play.T.w/2, G.play.T.y, G.CARD_W, G.CARD_H, base_card, center, {bypass_discovery_center = true, bypass_discovery_ui = true})
        
        -- Force re-application of the base card to mathematically guarantee the sprite atlas 
        -- is loaded, fixing the vanilla "blank art" rendering bug.
        if base_card and card.set_base then
            card:set_base(base_card)
        end
        
        local edition = self:get_random_edition()
        if edition then card:set_edition({[edition] = true}, true, true) end
        
        local seal = self:get_random_seal()
        if seal then card:set_seal(seal, true, true) end
        
        if G.hand.emplace then
            G.hand:emplace(card)
        else
            card.area = G.hand
            table.insert(G.hand.cards, card)
        end
    end
    
    -- Randomly select
    local num_selected = math.random(0, math.min(5, num_cards))
    for i=1, num_selected do
        local idx = math.random(1, #G.hand.cards)
        local c = G.hand.cards[idx]
        if not c.highlighted then
            c.highlighted = true
            G.hand.highlighted[#G.hand.highlighted + 1] = c
        end
    end
    
    G.hand:set_ranks()
    if G.hand.sort then G.hand:sort() end
end

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

function G.DATASET_MOD:calculate_boxes_for_card(card, draw_args)
    local boxes = {}
    local ox, oy = draw_args[6], draw_args[7]
    local cw, ch = (ox and ox > 0) and (ox * 2) or 71, (oy and oy > 0) and (oy * 2) or 95
    
    for box_name, box in pairs(self.config.boxes) do
        local center_x = cw/2 + box.ox * cw
        local center_y = ch/2 + box.oy * ch
        local hw = (box.w * cw) / 2
        local hh = (box.h * ch) / 2
        
        local tex_pts = {
            {center_x - hw, center_y - hh},
            {center_x + hw, center_y - hh},
            {center_x + hw, center_y + hh},
            {center_x - hw, center_y + hh}
        }
        
        local transformed_pts = {}
        for i, pt in ipairs(tex_pts) do
            local lx, ly = apply_local_transform(pt[1], pt[2], unpack(draw_args))
            local gx, gy = love.graphics.transformPoint(lx, ly)
            table.insert(transformed_pts, gx)
            table.insert(transformed_pts, gy)
        end
        
        boxes[box_name] = {
            points = transformed_pts,
            aabb = {
                math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
                math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]),
                math.max(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]) - math.min(transformed_pts[1], transformed_pts[3], transformed_pts[5], transformed_pts[7]),
                math.max(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8]) - math.min(transformed_pts[2], transformed_pts[4], transformed_pts[6], transformed_pts[8])
            }
        }
    end
    self.captured_boxes = self.captured_boxes or {}
    self.captured_boxes[card] = boxes
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

function G.DATASET_MOD:capture_frame()
    local roi = self.config.roi
    local rw, rh = math.max(1, math.floor(roi.w)), math.max(1, math.floor(roi.h))
    local rx, ry = math.floor(roi.x), math.floor(roi.y)
    
    local current_frame = self.frame_count
    
    love.graphics.captureScreenshot(function(img)
        local cropped = love.image.newImageData(rw, rh)
        cropped:paste(img, 0, 0, rx, ry, rw, rh)
        
        local filedata = cropped:encode("png")
        local raw_bytes = filedata:getString()
        
        local filename = "Mods/gamestate_dataset_cv/dataset/images/frame_" .. string.format("%04d", current_frame) .. ".png"
        love.filesystem.write(filename, raw_bytes)
        
        -- Explicitly release unmanaged C-heap memory to prevent gigabytes of leaks
        if filedata and filedata.release then filedata:release() end
        if cropped and cropped.release then cropped:release() end
        if img and img.release then img:release() end
    end)
    
    self:save_labels(current_frame)
    self.frame_count = self.frame_count + 1
    self.state = 'idle'
    self.capture_flash = 1.0
    self.capture_flash_text = "FRAME " .. current_frame .. " SAVED!"
    
    -- Periodically perform garbage collection sweep to prevent Lua heap bloat
    if self.frame_count % 25 == 0 then
        collectgarbage("collect")
    end
end

function G.DATASET_MOD:save_labels(frame_num)
    frame_num = frame_num or self.frame_count
    local roi = self.config.roi
    local yolo_labels = {}
    local json_cards = {}

    if G.hand and G.hand.cards then
        for idx, card in ipairs(G.hand.cards) do
            local boxes = self.captured_boxes and self.captured_boxes[card]
            local card_box = boxes and (boxes.card or boxes.state or boxes.identity)
            
            if card_box then
                local min_x, min_y, bw, bh = card_box.aabb[1], card_box.aabb[2], card_box.aabb[3], card_box.aabb[4]
                local clamped = {clamp_aabb_to_roi(roi, min_x, min_y, bw, bh)}
                if clamped[1] then
                    local nx, ny, nw, nh = to_yolo(roi, clamped[1], clamped[2], clamped[3], clamped[4])
                    
                    -- Stage 1 YOLO detection label: single class 0 ("card")
                    table.insert(yolo_labels, string.format("0 %.6f %.6f %.6f %.6f", nx, ny, nw, nh))
                    
                    -- Stage 2 Card Attributes
                    local rank = "None"
                    local suit = "None"
                    if card.config.center and card.config.center.name == "Stone Card" then
                        rank = "Stone"
                        suit = "None"
                    elseif card.base then
                        rank = tostring(card.base.value or "None")
                        suit = tostring(card.base.suit or "None")
                    end
                    
                    local edition = "None"
                    if card.edition then
                        if card.edition.foil then edition = "Foil"
                        elseif card.edition.holo then edition = "Holo"
                        elseif card.edition.polychrome then edition = "Polychrome"
                        elseif card.edition.negative then edition = "Negative"
                        end
                    end
                    
                    local enhancement = "None"
                    if card.config.center and card.config.center.name ~= "Default Base" and card.config.center.name ~= "Stone Card" then
                        enhancement = string.gsub(card.config.center.name, " Card", "")
                        enhancement = string.gsub(enhancement, " ", "")
                    elseif card.config.center and card.config.center.name == "Stone Card" then
                        enhancement = "Stone"
                    end
                    
                    local seal = card.seal or "None"
                    local selected = card.highlighted and true or false
                    
                    table.insert(json_cards, string.format(
                        '  {"card_index": %d, "box_yolo": [%.6f, %.6f, %.6f, %.6f], "box_pixel": [%.1f, %.1f, %.1f, %.1f], "rank": "%s", "suit": "%s", "edition": "%s", "enhancement": "%s", "seal": "%s", "selected": %s}',
                        idx, nx, ny, nw, nh, clamped[1] - roi.x, clamped[2] - roi.y, clamped[3], clamped[4],
                        rank, suit, edition, enhancement, seal, selected and "true" or "false"
                    ))
                end
            end
        end
    end

    -- Write Stage 1 YOLO detection label (class 0: card)
    local label_path = "Mods/gamestate_dataset_cv/dataset/labels/frame_" .. string.format("%04d", frame_num) .. ".txt"
    love.filesystem.write(label_path, table.concat(yolo_labels, "\n"))

    -- Write Stage 2 Structured JSON label with card attributes
    local json_path = "Mods/gamestate_dataset_cv/dataset/labels/frame_" .. string.format("%04d", frame_num) .. ".json"
    local json_content = "[\n" .. table.concat(json_cards, ",\n") .. "\n]\n"
    love.filesystem.write(json_path, json_content)
end

-- Hooks & Updates
function G.DATASET_MOD:update(dt)
    if self.capture_flash then
        self.capture_flash = math.max(0, self.capture_flash - dt)
    end
    
    if self.config.mode == 'setup' then
        local step = love.keyboard.isDown('lshift', 'rshift') and (0.5 * dt) or (0.1 * dt)
        local roi_step = love.keyboard.isDown('lshift', 'rshift') and (500 * dt) or (150 * dt)
        
        if self.selected_box == 'roi' then
            if love.keyboard.isDown('w') then self.config.roi.y = self.config.roi.y - roi_step end
            if love.keyboard.isDown('s') then self.config.roi.y = self.config.roi.y + roi_step end
            if love.keyboard.isDown('a') then self.config.roi.x = self.config.roi.x - roi_step end
            if love.keyboard.isDown('d') then self.config.roi.x = self.config.roi.x + roi_step end
            if love.keyboard.isDown('up') then self.config.roi.h = self.config.roi.h - roi_step end
            if love.keyboard.isDown('down') then self.config.roi.h = self.config.roi.h + roi_step end
            if love.keyboard.isDown('left') then self.config.roi.w = self.config.roi.w - roi_step end
            if love.keyboard.isDown('right') then self.config.roi.w = self.config.roi.w + roi_step end
        else
            local box = self.config.boxes[self.selected_box]
            if box then
                if love.keyboard.isDown('w') then box.oy = box.oy - step end
                if love.keyboard.isDown('s') then box.oy = box.oy + step end
                if love.keyboard.isDown('a') then box.ox = box.ox - step end
                if love.keyboard.isDown('d') then box.ox = box.ox + step end
                if love.keyboard.isDown('up') then box.h = box.h - step end
                if love.keyboard.isDown('down') then box.h = box.h + step end
                if love.keyboard.isDown('left') then box.w = box.w - step end
                if love.keyboard.isDown('right') then box.w = box.w + step end
            end
        end
    elseif self.config.mode == 'generation' then
        if self.state == 'idle' then
            self:generate_random_hand()
            self:randomize_background()
            
            if G.SETTINGS and G.SETTINGS.profile_events then
                G.SETTINGS.profile_events.high_contrast = (math.random() > 0.5)
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

function G.DATASET_MOD:draw_debug()
    if self.config.mode ~= 'setup' then return end

    love.graphics.push('all')
    -- Notice: We no longer call love.graphics.origin() here!
    -- This allows the boxes to inherit the game's exact Canvas->Window scaling matrix
    
    local roi = self.config.roi
    
    if self.selected_box == 'roi' then love.graphics.setColor(1, 1, 1, 1) else love.graphics.setColor(0, 1, 0, 1) end
    love.graphics.setLineWidth(self.selected_box == 'roi' and 4 or 2)
    love.graphics.rectangle('line', roi.x, roi.y, roi.w, roi.h)

    if G.hand and G.hand.cards then
        for _, card in ipairs(G.hand.cards) do
            local boxes = self.captured_boxes and self.captured_boxes[card]
            if boxes then
                for box_name, box in pairs(boxes) do
                    local pts = box.points
                    
                    local is_sel = (self.selected_box == box_name)
                    if box_name == 'card' then love.graphics.setColor(0, 1, 1, is_sel and 1 or 0.6)
                    else love.graphics.setColor(1, 1, 0, is_sel and 1 or 0.5)
                    end
                    
                    love.graphics.setLineWidth(is_sel and 4 or 2)
                    love.graphics.polygon('line', pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], pts[8])
                end
            end
        end
    end
    
    -- Draw UI Text
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print("DATASET CALIBRATION MODE", 10, 10)
    love.graphics.print("Currently Editing: " .. self.selected_box, 10, 30)
    love.graphics.print("1: ROI | 2: Card Box | W,A,S,D: Move Offset | Arrows: Adjust Size", 10, 50)
    love.graphics.print("TAB: Start Generation | R: Randomize Hand | T: Toggle Selected | C: Capture Test Frame", 10, 70)
    love.graphics.print("[, ]: Save / Load Config", 10, 90)
    
    if self.capture_flash and self.capture_flash > 0 then
        love.graphics.setColor(0, 1, 0, self.capture_flash)
        love.graphics.print("IMAGE SAVED!", 10, 110)
    end
    
    love.graphics.pop()
end

function G.DATASET_MOD:on_keypressed(key)
    if key == 'tab' then
        if self.config.mode == 'setup' then
            self.config.mode = 'generation'
            self.state = 'idle'
            print("DATASET: Generation Mode Started")
        else
            self.config.mode = 'setup'
            if G.SETTINGS and G.SETTINGS.profile_events then
                G.SETTINGS.profile_events.high_contrast = false
            end
            print("DATASET: Setup Mode Started")
        end
    elseif self.config.mode == 'setup' then
        if key == 'r' then
            self:generate_random_hand()
        elseif key == 't' then
            if G.hand and G.hand.cards then
                -- Deselect all
                for _, c in ipairs(G.hand.cards) do
                    c.highlighted = false
                end
                G.hand.highlighted = {}
                
                -- Randomly select 0 to 5
                local num_selected = math.random(0, math.min(5, #G.hand.cards))
                for i=1, num_selected do
                    local idx = math.random(1, #G.hand.cards)
                    local c = G.hand.cards[idx]
                    if not c.highlighted then
                        c.highlighted = true
                        G.hand.highlighted[#G.hand.highlighted + 1] = c
                    end
                end
            end
        elseif key == 'c' then
            self:capture_frame()
        elseif key == ',' then
            self:save_config()
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG SAVED!"
        elseif key == '.' then
            self:load_config()
            self.capture_flash = 1.0
            self.capture_flash_text = "CONFIG LOADED!"
        elseif key == '1' then self.selected_box = 'roi'
        elseif key == '2' then self.selected_box = 'card'
        end
    end
end

local orig_lg_draw = love.graphics.draw
function love.graphics.draw(drawable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    local is_front = false
    local card = nil
    if G.DATASET_MOD and G.DATASET_MOD.current_card then
        card = G.DATASET_MOD.current_card
        local cr, cg, cb, ca = love.graphics.getColor()
        -- Skip the shadow, which is drawn with low alpha
        if ca > 0.9 and (cr > 0.1 or cg > 0.1 or cb > 0.1) then
            is_front = true
            G.DATASET_MOD.captured_boxes = G.DATASET_MOD.captured_boxes or {}
            if not G.DATASET_MOD.captured_boxes[card] then
                local draw_args = {}
                if type(arg1) == "userdata" and arg1.typeOf and arg1:typeOf("Quad") then
                    draw_args = {arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10}
                else
                    draw_args = {arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9}
                end
                G.DATASET_MOD:calculate_boxes_for_card(card, draw_args)
            end
        end
    end
    
    orig_lg_draw(drawable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    
    if is_front and G.DATASET_MOD.config.mode == 'setup' then
        local quad, x, y, r, sx, sy, ox, oy, kx, ky
        if type(arg1) == "userdata" and arg1.typeOf and arg1:typeOf("Quad") then
            x, y, r, sx, sy, ox, oy, kx, ky = arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
        else
            x, y, r, sx, sy, ox, oy, kx, ky = arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9
        end
        
        love.graphics.push('all')
        x = x or 0; y = y or 0; r = r or 0; sx = sx or 1; sy = sy or 1; ox = ox or 0; oy = oy or 0
        love.graphics.translate(x, y)
        love.graphics.rotate(r)
        love.graphics.scale(sx, sy)
        love.graphics.translate(-ox, -oy)
        
        -- Dynamically calculate card size from its rotational origin
        local cw, ch = (ox and ox > 0) and (ox * 2) or 71, (oy and oy > 0) and (oy * 2) or 95
        
        local current_shader = love.graphics.getShader()
        love.graphics.setShader()
        
        local function draw_box(box_name, box)
            local is_sel = (G.DATASET_MOD.selected_box == box_name)
            if box_name == 'identity' then love.graphics.setColor(1, 0, 0, is_sel and 1 or 0.5)
            elseif box_name == 'edition' then love.graphics.setColor(0, 1, 1, is_sel and 1 or 0.5)
            elseif box_name == 'enhancement' then love.graphics.setColor(1, 0, 1, is_sel and 1 or 0.5)
            elseif box_name == 'seal' then love.graphics.setColor(1, 1, 0, is_sel and 1 or 0.5)
            elseif box_name == 'state' then love.graphics.setColor(1, 0.5, 0, is_sel and 1 or 0.5)
            end
            
            love.graphics.setLineWidth(is_sel and 1.5 or 0.8)
            
            local center_x = cw/2 + box.ox * cw
            local center_y = ch/2 + box.oy * ch
            local hw = (box.w * cw) / 2
            local hh = (box.h * ch) / 2
            
            love.graphics.polygon('line', 
                center_x - hw, center_y - hh,
                center_x + hw, center_y - hh,
                center_x + hw, center_y + hh,
                center_x - hw, center_y + hh
            )
        end
        
        -- Draw inactive boxes first
        for box_name, box in pairs(G.DATASET_MOD.config.boxes) do
            if G.DATASET_MOD.selected_box ~= box_name then
                draw_box(box_name, box)
            end
        end
        
        -- Draw active box on top
        if G.DATASET_MOD.config.boxes[G.DATASET_MOD.selected_box] then
            draw_box(G.DATASET_MOD.selected_box, G.DATASET_MOD.config.boxes[G.DATASET_MOD.selected_box])
        end
        
        love.graphics.setShader(current_shader)
        love.graphics.pop()
    end
end

local orig_card_draw = Card.draw
function Card:draw(layer)
    if G.DATASET_MOD then
        local in_hand = false
        if G.hand and G.hand.cards then
            for _, c in ipairs(G.hand.cards) do
                if c == self then in_hand = true; break end
            end
        end
        if in_hand then
            G.DATASET_MOD.current_card = self
            G.DATASET_MOD.captured_boxes = G.DATASET_MOD.captured_boxes or {}
            G.DATASET_MOD.captured_boxes[self] = nil
        else
            G.DATASET_MOD.current_card = nil
        end
    end
    orig_card_draw(self, layer)
    if G.DATASET_MOD then
        G.DATASET_MOD.current_card = nil
    end
end

local orig_love_update = love.update
function love.update(dt)
    if orig_love_update then orig_love_update(dt) end
    if G.DATASET_MOD then G.DATASET_MOD:update(dt) end
end

local orig_love_draw = love.draw
function love.draw()
    if orig_love_draw then orig_love_draw() end
    if G.DATASET_MOD and G.DATASET_MOD.config.mode == 'setup' then
        love.graphics.push('all')
        local roi = G.DATASET_MOD.config.roi
        love.graphics.setColor(1, 1, 1, 1)
        love.graphics.setLineWidth(G.DATASET_MOD.selected_box == 'roi' and 4 or 2)
        love.graphics.rectangle('line', roi.x, roi.y, roi.w, roi.h)
        
        -- Draw UI Text
        love.graphics.setColor(1, 1, 1, 1)
        love.graphics.print("DATASET CALIBRATION MODE", 10, 10)
        love.graphics.print("Currently Editing: " .. G.DATASET_MOD.selected_box, 10, 30)
        love.graphics.print("1-6: Select Box | W,A,S,D: Move Offset | Arrows: Adjust Size", 10, 50)
        love.graphics.print("TAB: Start Generation | R: Randomize Hand | T: Toggle Selected | C: Capture Test Frame", 10, 70)
        love.graphics.print(",: Save Config | .: Load Config", 10, 90)
        love.graphics.print(string.format("Next Frame: %04d | Lua Memory: %.1f MB", G.DATASET_MOD.frame_count, collectgarbage("count") / 1024), 10, 110)
        
        if G.DATASET_MOD.capture_flash and G.DATASET_MOD.capture_flash > 0 then
            love.graphics.setColor(0, 1, 0, G.DATASET_MOD.capture_flash)
            love.graphics.print(G.DATASET_MOD.capture_flash_text or "SAVED!", 10, 130)
        end
        love.graphics.pop()
    end
end

local orig_love_keypressed = love.keypressed
function love.keypressed(key)
    if orig_love_keypressed then orig_love_keypressed(key) end
    if G.DATASET_MOD then G.DATASET_MOD:on_keypressed(key) end
end
