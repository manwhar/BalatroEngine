-- gamestate_dataset_cv/main.lua
-- Root orchestrator for GameState Dataset CV mod.

G.DATASET_MOD = {
    state = 'idle',
    timer = 0,
    capture_flash = 0,
    capture_flash_text = "",
    captured_boxes = setmetatable({}, {__mode = "k"}),
    classes = {},
    class_mapping = {},
    next_class_id = 0,
    mod_path = "",
    current_card = nil
}

-- Determine mod root directory
local function get_mod_path()
    local p = love.filesystem.getSaveDirectory() .. "/Mods/gamestate_dataset_cv/"
    if SMODS and SMODS.current_mod and SMODS.current_mod.path then
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

-- Submodule Loader
local function load_submodule(rel_path)
    -- 1. Try SMODS.load_file
    if SMODS and SMODS.load_file then
        local chunk = SMODS.load_file(rel_path, "gamestate_dataset_cv")
        if chunk then return chunk() end
        local chunk2 = SMODS.load_file(rel_path)
        if chunk2 then return chunk2() end
    end
    -- 2. Try NFS.read
    if NFS and NFS.read and G.DATASET_MOD and G.DATASET_MOD.mod_path then
        local full_p = G.DATASET_MOD.mod_path .. rel_path
        local content = NFS.read(full_p)
        if content then
            local chunk = (loadstring or load)(content, "=[GDCV " .. rel_path .. "]")
            if chunk then return chunk() end
        end
    end
    -- 3. Fallback to love.filesystem
    local candidates = {
        "Mods/gamestate_dataset_cv/" .. rel_path,
        rel_path
    }
    for _, path in ipairs(candidates) do
        if love.filesystem.getInfo(path) then
            local content = love.filesystem.read(path)
            local chunk = (loadstring or load)(content, "=[GDCV " .. path .. "]")
            if chunk then return chunk() end
        end
    end
    error("GDCV: Failed to load module " .. tostring(rel_path))
end

-- Load Modular Components
local ConfigModule   = load_submodule("src/config.lua")
local CardsGenModule = load_submodule("src/cards_gen.lua")
local ShopGenModule  = load_submodule("src/shop_gen.lua")
local CaptureModule  = load_submodule("src/capture.lua")
local UIModule       = load_submodule("src/ui.lua")

-- Initialize configuration, counters, and directories
ConfigModule.init(G.DATASET_MOD)

-- Initialize SMODS callbacks and config tab
UIModule.init_callbacks(G.DATASET_MOD)
if SMODS and SMODS.current_mod then
    SMODS.current_mod.config_tab = UIModule.create_config_tab
end
if SMODS and SMODS.Mods and SMODS.Mods["gamestate_dataset_cv"] then
    SMODS.Mods["gamestate_dataset_cv"].config_tab = UIModule.create_config_tab
end

-- Backward compatibility helpers for classes.txt
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
        end
    end
end

function G.DATASET_MOD:save_classes()
    local ordered = {}
    for name, id in pairs(self.class_mapping) do
        ordered[id] = name
    end
    local lines = {}
    for i = 0, self.next_class_id - 1 do
        table.insert(lines, ordered[i])
    end
    love.filesystem.write("Mods/gamestate_dataset_cv/dataset/classes.txt", table.concat(lines, "\n"))
end

function G.DATASET_MOD:get_class_id(class_name)
    if not self.class_mapping[class_name] then
        self.class_mapping[class_name] = self.next_class_id
        self.next_class_id = self.next_class_id + 1
        self:save_classes()
    end
    return self.class_mapping[class_name]
end

G.DATASET_MOD:load_classes()

-- ==========================================
-- ENGINE HOOKS
-- ==========================================

-- 1. love.graphics.draw hook: compute bounding boxes on card front draw
local orig_lg_draw = love.graphics.draw
function love.graphics.draw(drawable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    local is_front = false
    local card = nil
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' and G.DATASET_MOD.current_card then
        card = G.DATASET_MOD.current_card
        local cr, cg, cb, ca = love.graphics.getColor()
        -- Skip the shadow quad (drawn with low alpha or dark colour)
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
end

-- 2. Card:draw hook: mark card as tracked if inside active areas
local orig_card_draw = Card.draw
function Card:draw(layer)
    if G.DATASET_MOD and G.DATASET_MOD.config and G.DATASET_MOD.config.mode ~= 'off' then
        local tracked = false
        local mode = G.DATASET_MOD.config.mode

        if mode == 'card_classifier' then
            if G.hand and G.hand.cards then
                for _, c in ipairs(G.hand.cards) do
                    if c == self then tracked = true; break end
                end
            end
        elseif mode == 'shop_classifier' then
            if G.jokers and G.jokers.cards then
                for _, c in ipairs(G.jokers.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.shop_jokers and G.shop_jokers.cards then
                for _, c in ipairs(G.shop_jokers.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.consumeables and G.consumeables.cards then
                for _, c in ipairs(G.consumeables.cards) do
                    if c == self then tracked = true; break end
                end
            end
            if not tracked and G.pack_cards and G.pack_cards.cards then
                for _, c in ipairs(G.pack_cards.cards) do
                    if c == self then tracked = true; break end
                end
            end
        end

        if tracked then
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

-- 3. love.update hook
local orig_love_update = love.update
function love.update(dt)
    if orig_love_update then orig_love_update(dt) end
    if G.DATASET_MOD then G.DATASET_MOD:update(dt) end
end

-- 4. love.draw hook
local orig_love_draw = love.draw
function love.draw()
    if orig_love_draw then orig_love_draw() end
    if G.DATASET_MOD then G.DATASET_MOD:draw_debug() end
end

-- 5. love.keypressed hook
local orig_love_keypressed = love.keypressed
function love.keypressed(key)
    if orig_love_keypressed then orig_love_keypressed(key) end
    if G.DATASET_MOD then G.DATASET_MOD:on_keypressed(key) end
end

print("DATASET: GameState Dataset CV loaded successfully!")
