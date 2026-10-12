-- gamestate_dataset_cv/src/config.lua

local Config = {}

local DEFAULT_CONFIG = {
    mode = 'off', -- 'off', 'card_classifier', 'shop_classifier'
    calibration_mode = 'setup', -- 'setup' or 'generation'
    max_frames = 10000,
    frames_per_phase = 100, -- 100 shop screens, then 100 pack screens
    perkeo_max_consumables = 10, -- Max consumables in perkeo mode (configurable)
    rois = {
        card_hand    = {x = 508.319934, y = 567.839093, w = 1071.056521, h = 322.734447},
        joker_bar    = {x = 320.0,      y = 10.0,       w = 1280.0,      h = 260.0},
        shop         = {x = 360.0,      y = 140.0,      w = 1200.0,      h = 380.0},
        pack         = {x = 300.0,      y = 260.0,      w = 1320.0,      h = 500.0},
        consumeables = {x = 1250.0,     y = 10.0,       w = 420.0,       h = 260.0},
    },
    boxes = {
        card             = {w = 0.465890, h = 1.040970, ox = -0.240265, oy = 0.020829},
        joker_bar_card   = {w = 0.550000, h = 1.000000, ox = -0.225000, oy = 0.000000},
        item_card        = {w = 1.000000, h = 1.000000, ox = 0.000000,  oy = 0.000000},
        consumeable_card = {w = 1.000000, h = 1.000000, ox = 0.000000,  oy = 0.000000},
    }
}

function Config.init(mod)
    mod.config = {
        mode = DEFAULT_CONFIG.mode,
        calibration_mode = DEFAULT_CONFIG.calibration_mode,
        max_frames = DEFAULT_CONFIG.max_frames,
        frames_per_phase = DEFAULT_CONFIG.frames_per_phase,
        perkeo_max_consumables = DEFAULT_CONFIG.perkeo_max_consumables,
        rois = {},
        boxes = {}
    }

    for k, v in pairs(DEFAULT_CONFIG.rois) do
        mod.config.rois[k] = {x = v.x, y = v.y, w = v.w, h = v.h}
    end
    for k, v in pairs(DEFAULT_CONFIG.boxes) do
        mod.config.boxes[k] = {w = v.w, h = v.h, ox = v.ox, oy = v.oy}
    end

    mod.selected_target = 'roi_joker_bar'
    mod.frame_counts = {card = 0, shop = 0}

    Config.init_directories()
    mod:load_config()
    Config.init_frame_counts(mod)
    Config.init_classes(mod)
end

function Config.init_directories()
    local dirs = {
        "Mods/gamestate_dataset_cv/dataset",
        "Mods/gamestate_dataset_cv/dataset/images",
        "Mods/gamestate_dataset_cv/dataset/labels",
        "Mods/gamestate_dataset_cv/dataset_shop",
        "Mods/gamestate_dataset_cv/dataset_shop/images",
        "Mods/gamestate_dataset_cv/dataset_shop/labels"
    }
    for _, dir in ipairs(dirs) do
        love.filesystem.createDirectory(dir)
    end
end

function Config.init_frame_counts(mod)
    -- 1. Card dataset frame counter
    local max_card_idx = -1
    local card_dirs = {"Mods/gamestate_dataset_cv/dataset/images", "Mods/gamestate_dataset_cv/dataset/labels"}
    for _, dir in ipairs(card_dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local num = f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_card_idx then max_card_idx = n end
                end
            end
        end
    end
    mod.frame_counts.card = max_card_idx + 1

    -- 2. Shop dataset frame counter
    local max_shop_idx = -1
    local shop_dirs = {"Mods/gamestate_dataset_cv/dataset_shop/images", "Mods/gamestate_dataset_cv/dataset_shop/labels"}
    for _, dir in ipairs(shop_dirs) do
        if love.filesystem.getInfo(dir) then
            local files = love.filesystem.getDirectoryItems(dir)
            for _, f in ipairs(files) do
                local num = f:match("^frame_(%d+)_%w+%.%w+$") or f:match("^frame_(%d+)%.%w+$")
                if num then
                    local n = tonumber(num)
                    if n and n > max_shop_idx then max_shop_idx = n end
                end
            end
        end
    end
    mod.frame_counts.shop = max_shop_idx + 1
    mod.frame_count = (mod.config.mode == 'shop_classifier') and mod.frame_counts.shop or mod.frame_counts.card
    print(string.format("DATASET: Initialized counters - Card: %d, Shop: %d", mod.frame_counts.card, mod.frame_counts.shop))
end

function Config.init_classes(mod)
    local shop_classes_path = "Mods/gamestate_dataset_cv/dataset_shop/classes.txt"
    if not love.filesystem.getInfo(shop_classes_path) then
        local shop_classes = "Joker\nTarot\nSpectral\nPlanet\n"
        love.filesystem.write(shop_classes_path, shop_classes)
    end
end

function G.DATASET_MOD:save_config()
    local content = "return {\n"
    content = content .. string.format("  mode = %q,\n", self.config.mode or 'off')
    content = content .. string.format("  calibration_mode = %q,\n", self.config.calibration_mode or 'setup')
    content = content .. string.format("  max_frames = %d,\n", self.config.max_frames or 10000)
    content = content .. string.format("  frames_per_phase = %d,\n", self.config.frames_per_phase or 100)
    content = content .. string.format("  perkeo_max_consumables = %d,\n", self.config.perkeo_max_consumables or 10)
    
    content = content .. "  rois = {\n"
    for k, v in pairs(self.config.rois) do
        content = content .. string.format("    %s = {x=%f, y=%f, w=%f, h=%f},\n", k, v.x, v.y, v.w, v.h)
    end
    content = content .. "  },\n"

    content = content .. "  boxes = {\n"
    for k, v in pairs(self.config.boxes) do
        content = content .. string.format("    %s = {w=%f, h=%f, ox=%f, oy=%f},\n", k, v.w, v.h, v.ox, v.oy)
    end
    content = content .. "  }\n}\n"

    love.filesystem.createDirectory("Mods/gamestate_dataset_cv/dataset")
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/boxes.lua", content)
    print("DATASET: Config saved to boxes.lua!")
end

function G.DATASET_MOD:load_config()
    local path = "Mods/gamestate_dataset_cv/dataset/boxes.lua"
    if love.filesystem.getInfo(path) then
        local content = love.filesystem.read(path)
        local chunk = (loadstring or load)(content)
        if chunk then
            local t = chunk()
            if type(t) == "table" then
                if t.mode then self.config.mode = t.mode end
                if t.calibration_mode then self.config.calibration_mode = t.calibration_mode end
                if t.max_frames then self.config.max_frames = tonumber(t.max_frames) or 10000 end
                if t.frames_per_phase then self.config.frames_per_phase = tonumber(t.frames_per_phase) or 100 end
                if t.perkeo_max_consumables then self.config.perkeo_max_consumables = tonumber(t.perkeo_max_consumables) or 10 end

                -- Backwards compatibility: single 'roi' table
                if t.roi and type(t.roi) == "table" and not t.rois then
                    self.config.rois.card_hand = {
                        x = tonumber(t.roi.x) or DEFAULT_CONFIG.rois.card_hand.x,
                        y = tonumber(t.roi.y) or DEFAULT_CONFIG.rois.card_hand.y,
                        w = tonumber(t.roi.w) or DEFAULT_CONFIG.rois.card_hand.w,
                        h = tonumber(t.roi.h) or DEFAULT_CONFIG.rois.card_hand.h
                    }
                end

                if t.rois and type(t.rois) == "table" then
                    for k, v in pairs(t.rois) do
                        if type(v) == "table" then
                            self.config.rois[k] = {
                                x = tonumber(v.x) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].x or 0),
                                y = tonumber(v.y) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].y or 0),
                                w = tonumber(v.w) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].w or 100),
                                h = tonumber(v.h) or (DEFAULT_CONFIG.rois[k] and DEFAULT_CONFIG.rois[k].h or 100)
                            }
                        end
                    end
                end

                if t.boxes and type(t.boxes) == "table" then
                    for k, v in pairs(t.boxes) do
                        if type(v) == "table" then
                            self.config.boxes[k] = {
                                w = tonumber(v.w) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].w or 1),
                                h = tonumber(v.h) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].h or 1),
                                ox = tonumber(v.ox) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].ox or 0),
                                oy = tonumber(v.oy) or (DEFAULT_CONFIG.boxes[k] and DEFAULT_CONFIG.boxes[k].oy or 0)
                            }
                        end
                    end
                end
                print("DATASET: Config loaded from boxes.lua!")
            end
        end
    end
end

function G.DATASET_MOD:set_mode(mode)
    if mode == 'card_classifier' or mode == 'shop_classifier' or mode == 'off' then
        self.config.mode = mode
        if mode == 'card_classifier' then
            self.selected_target = 'roi_card_hand'
            self.frame_count = self.frame_counts.card
        elseif mode == 'shop_classifier' then
            self.selected_target = 'roi_joker_bar'
            self.frame_count = self.frame_counts.shop
        end
        print("DATASET: Mode set to " .. tostring(mode))
    end
end

return Config
