-- gamestate_dataset_cv/src/end_round_gen.lua
-- Scenario randomizer and safe in-place blind reset loop for the temporal end-round dataset mode.

local EndRoundGen = {}

-- Weighted scenario archetypes matching both typical gameplay and adversarial boundary cases.
local ARCHETYPE_WEIGHTS = {
    { key = 'fast_clean',            weight = 0.22 },
    { key = 'heavy_retrigger',       weight = 0.28 },
    { key = 'all_glass_shatter',     weight = 0.15 },
    { key = 'partial_glass_shatter', weight = 0.10 },
    { key = 'held_end_round',        weight = 0.15 },
    { key = 'end_round_jokers',      weight = 0.10 },
}

local GAMESPEED_WEIGHTS = {
    { key = 0.5, weight = 0.08 },
    { key = 1,   weight = 0.27 },
    { key = 2,   weight = 0.30 },
    { key = 4,   weight = 0.35 },
}

local CRT_WEIGHTS = {
    { key = 0,   weight = 0.22 },
    { key = 35,  weight = 0.25 },
    { key = 70,  weight = 0.33 },
    { key = 100, weight = 0.20 },
}

-- Specialty joker pools targeted per scenario archetype.
local RETRIGGER_JOKERS = {
    'j_mime', 'j_sock_and_buskin', 'j_hack', 'j_hanging_chad',
    'j_dusk', 'j_selzer', 'j_blueprint', 'j_brainstorm',
}

local END_ROUND_JOKERS = {
    'j_golden', 'j_cloud_9', 'j_rocket', 'j_satellite', 'j_todo_list',
    'j_egg', 'j_gift', 'j_popcorn', 'j_turtle_bean', 'j_gros_michel',
    'j_cavendish', 'j_campfire', 'j_invisible', 'j_hit_the_road',
    'j_delayed_grat',
}

local SEAL_CHOICES = { 'Gold', 'Red', 'Blue', 'Purple' }
local ENHANCEMENT_CHOICES = {
    'c_base', 'm_bonus', 'm_mult', 'm_wild', 'm_glass',
    'm_steel', 'm_stone', 'm_gold', 'm_lucky',
}

--- Sample an item from a `{ { key = ..., weight = ... }, ... }` table.
local function weighted_choice(entries)
    local total = 0
    for _, entry in ipairs(entries) do
        total = total + entry.weight
    end
    local r = math.random() * total
    local acc = 0
    for _, entry in ipairs(entries) do
        acc = acc + entry.weight
        if r <= acc then
            return entry.key
        end
    end
    return entries[#entries].key
end

--- Snap a Moveable (and optional area cards) straight to target T without easing lag.
local function snap_cardarea(area)
    if not area then return end
    if area.set_ranks then area:set_ranks() end
    if area.align_cards then area:align_cards() end
    if area.hard_set_VT then area:hard_set_VT() end
    if area.cards then
        for _, c in ipairs(area.cards) do
            if c.hard_set_VT then c:hard_set_VT() end
        end
    end
end

--- Remove a list of floating engine objects safely if not already removed.
local function remove_matching_objects(tbl, predicate)
    if type(tbl) ~= 'table' then return end
    for i = #tbl, 1, -1 do
        local obj = tbl[i]
        if obj and not obj.REMOVED and predicate(obj) then
            pcall(function() obj:remove() end)
        end
    end
end

local function shallow_copy(tbl)
    if type(tbl) ~= 'table' then return nil end
    local copy = {}
    for k, v in pairs(tbl) do copy[k] = v end
    return copy
end

--- Pick a deterministic matching `G.P_CARDS` center given optional suit and rank suffixes (e.g. 'H', 'A').
local function find_p_card(suit_prefix, rank_suffix)
    if not G.P_CARDS then return nil end
    if suit_prefix and rank_suffix then
        return G.P_CARDS[suit_prefix .. '_' .. rank_suffix]
    end
    local matches = {}
    for k, v in pairs(G.P_CARDS) do
        if k ~= 'empty' and type(v) == 'table' and v.suit and v.value then
            if (not suit_prefix or k:sub(1, 2) == (suit_prefix .. '_'))
                and (not rank_suffix or k:sub(-#rank_suffix) == rank_suffix) then
                matches[#matches + 1] = v
            end
        end
    end
    return (#matches > 0) and matches[math.random(#matches)] or nil
end

--- Save user settings and run-protection flags before starting end-round clip generation.
function EndRoundGen.backup_user_state(mod)
    if mod._end_round_backup then return end

    local gfx = G.SETTINGS and G.SETTINGS.GRAPHICS or {}
    local snd = G.SETTINGS and G.SETTINGS.SOUND or {}
    local cd = G.SETTINGS and G.SETTINGS.CUSTOM_DECK or {}
    mod._end_round_backup = {
        gamespeed = G.SETTINGS and G.SETTINGS.GAMESPEED or 1,
        crt = gfx.crt,
        bloom = gfx.bloom,
        reduced_motion = G.SETTINGS and G.SETTINGS.reduced_motion or false,
        tutorial_complete = G.SETTINGS and G.SETTINGS.tutorial_complete,
        volume = snd.volume,
        collabs = shallow_copy(cd.Collabs),
        colour_palettes = shallow_copy(G.SETTINGS and G.SETTINGS.colour_palettes),
        no_saving = G.F_NO_SAVING,
        no_achievements = G.F_NO_ACHIEVEMENTS,
        win_ante = G.GAME and G.GAME.win_ante,
        seeded = G.GAME and G.GAME.seeded,
        facing_blind = G.GAME and G.GAME.facing_blind,
        prob_normal = G.GAME and G.GAME.probabilities and G.GAME.probabilities.normal,
        cursor_visible = G.CURSOR and G.CURSOR.states and G.CURSOR.states.visible,
        hand_limit = G.hand and G.hand.config and G.hand.config.card_limit,
        joker_limit = G.jokers and G.jokers.config and G.jokers.config.card_limit,
        jd_saved = false,
    }
    if mod.save_joker_display_config then
        mod:save_joker_display_config()
        mod._end_round_backup.jd_saved = true
    end

    -- Prevent auto-saves, profile/career stat disk writes, unlocks/achievements, and win modals.
    G.F_NO_SAVING = true
    G.F_NO_ACHIEVEMENTS = true
    if G.SETTINGS then G.SETTINGS.tutorial_complete = true end
    if G.GAME then
        G.GAME.win_ante = 999
        G.GAME.seeded = true
    end
    if G.SETTINGS and G.SETTINGS.SOUND then G.SETTINGS.SOUND.volume = 0 end
end

--- Restore all user settings and clean up active clip state when exiting generation mode.
function EndRoundGen.restore_user_state(mod)
    local b = mod._end_round_backup
    if not b then return end

    if G.SETTINGS then
        G.SETTINGS.GAMESPEED = b.gamespeed or 1
        G.SETTINGS.reduced_motion = b.reduced_motion
        if b.tutorial_complete ~= nil then G.SETTINGS.tutorial_complete = b.tutorial_complete end
        if G.SETTINGS.GRAPHICS then
            if b.crt ~= nil then G.SETTINGS.GRAPHICS.crt = b.crt end
            if b.bloom ~= nil then G.SETTINGS.GRAPHICS.bloom = b.bloom end
        end
        if G.SETTINGS.SOUND and b.volume ~= nil then
            G.SETTINGS.SOUND.volume = b.volume
        end
        if b.colour_palettes then
            G.SETTINGS.colour_palettes = b.colour_palettes
        end
        if b.collabs and G.SETTINGS.CUSTOM_DECK then
            G.SETTINGS.CUSTOM_DECK.Collabs = b.collabs
            if G.FUNCS and G.FUNCS.update_suit_colours then
                for suit, skin in pairs(b.collabs) do
                    pcall(G.FUNCS.update_suit_colours, suit, skin)
                end
            end
        end
    end

    if b.no_saving ~= nil then G.F_NO_SAVING = b.no_saving end
    if b.no_achievements ~= nil then G.F_NO_ACHIEVEMENTS = b.no_achievements end
    if G.GAME then
        if b.win_ante ~= nil then G.GAME.win_ante = b.win_ante end
        if b.seeded ~= nil then G.GAME.seeded = b.seeded end
        if b.facing_blind ~= nil then G.GAME.facing_blind = b.facing_blind end
        if b.prob_normal ~= nil and G.GAME.probabilities then
            G.GAME.probabilities.normal = b.prob_normal
        end
    end
    if G.CURSOR and G.CURSOR.states and b.cursor_visible ~= nil then
        G.CURSOR.states.visible = b.cursor_visible
    end
    if G.hand and G.hand.config and b.hand_limit ~= nil then
        G.hand.config.card_limit = b.hand_limit
    end
    if G.jokers and G.jokers.config and b.joker_limit ~= nil then
        G.jokers.config.card_limit = b.joker_limit
    end
    if b.jd_saved and mod.restore_joker_display_config then
        mod:restore_joker_display_config()
    end

    mod._end_round_backup = nil
end

--- Destroy transient scoring particles, floating attention text, and round-eval overlays.
function EndRoundGen.sweep_transient_effects(mod)
    if mod.ShopGen and mod.ShopGen.clear_all_event_queues then
        mod.ShopGen.clear_all_event_queues()
    end

    if G.round_eval then
        pcall(function() G.round_eval:remove() end)
        G.round_eval = nil
    end
    if G.achievement_notification then
        pcall(function() G.achievement_notification:remove() end)
        G.achievement_notification = nil
    end

    if G.I then
        local protected_uibox = {
            [G.HUD or false] = true,
            [G.HUD_blind or false] = true,
            [G.HUD_tags or false] = true,
            [G.buttons or false] = true,
            [G.OVERLAY_MENU or false] = true,
            [G.shop or false] = true,
        }

        -- Remove attention text and floating card/joker eval popups spawned during scoring/end_round.
        remove_matching_objects(G.I.UIBOX, function(uib)
            if protected_uibox[uib] then return false end
            if uib.attention_text then return true end
            if not uib.parent and uib.role and uib.role.major then
                local maj = uib.role.major
                if maj.REMOVED or (maj.is and maj:is(Card)) then
                    return true
                end
            end
            return false
        end)

        remove_matching_objects(G.I.MOVEABLE, function(m)
            return m.is and Particles and m:is(Particles)
        end)

        remove_matching_objects(G.I.POPUP, function(_) return true end)
    end
end

--- Reset board CardAreas, game state, physics, and GC between clips so memory stays flat.
function EndRoundGen.cleanup_clip_state(mod)
    if not G.GAME then return end

    EndRoundGen.sweep_transient_effects(mod)

    local clear_area = mod.ShopGen and mod.ShopGen.clear_card_area
    if clear_area then
        -- Remove Jokers before G.hand so Card:remove_from_deck() hand-size adjustments apply cleanly.
        clear_area(G.jokers)
        clear_area(G.consumeables)
        clear_area(G.play)
        clear_area(G.hand)
        clear_area(G.discard)
        clear_area(G.deck)
    end
    G.playing_cards = {}

    if mod.ShopGen and mod.ShopGen.cleanup_cycle then
        mod.ShopGen.cleanup_cycle()
    else
        collectgarbage("collect")
        collectgarbage("collect")
    end

    G.ACC = 0
    G.ACC_state = G.STATES.SELECTING_HAND
    G.RESET_JIGGLES = nil
    G.VIBRATION = 0
    G.SETTINGS.paused = false
    if G.ROOM then G.ROOM.jiggle = 0 end

    G.GAME.STOP_USE = 0
    G.GAME.dollar_buffer = 0
    G.GAME.consumeable_buffer = 0
    G.GAME.joker_buffer = 0
    G.GAME.pack_choices = nil
    G.GAME.won = false
    G.GAME.win_ante = 999
    G.GAME.seeded = true
    G.GAME.facing_blind = true
    G.F_NO_ACHIEVEMENTS = true
    G.GAME.probabilities = G.GAME.probabilities or { normal = 1 }
    G.GAME.probabilities.normal = 1

    if G.GAME.current_round then
        G.GAME.current_round.hands_played = 0
        G.GAME.current_round.discards_used = 0
        G.GAME.current_round.jokers_purchased = 0
        G.GAME.current_round.most_played_poker_hand = G.GAME.current_round.most_played_poker_hand or 'High Card'
    end
    if G.GAME.hands then
        for _, h in pairs(G.GAME.hands) do
            h.played_this_round = 0
            h.level = math.max(1, h.level or 1)
            h.chips = math.max(5, h.chips or 5)
            h.mult = math.max(1, h.mult or 1)
        end
    end
    if G.GAME.blind then
        G.GAME.blind.block_play = nil
        G.GAME.blind.triggered = false
        G.GAME.blind.prepped = nil
        G.GAME.blind.dissolve = nil
        G.GAME.blind.hands = {}
        G.GAME.blind.only_hand = nil
        if G.GAME.blind.children and G.GAME.blind.children.particles then
            pcall(function() G.GAME.blind.children.particles:remove() end)
            G.GAME.blind.children.particles = nil
        end
    end
    if G.CONTROLLER then
        G.CONTROLLER.locks = {}
        G.CONTROLLER.dragging.target = nil
    end

    if G.hand and G.hand.config then
        G.hand.config.card_limit = 8
        G.hand.config.highlighted_limit = 5
    end
    if G.jokers and G.jokers.config then
        G.jokers.config.card_limit = 5
    end
    if G.consumeables and G.consumeables.config then
        G.consumeables.config.card_limit = 2
    end

    G.STATE = G.STATES.SELECTING_HAND
    G.STATE_COMPLETE = true
end

--- Choose a blind center compatible with the current scenario archetype.
local function pick_blind_for_archetype(archetype)
    local bl = G.P_BLINDS or {}
    -- Small and Big blinds are common (~40%), Boss blinds cover flipped/debuffed/UI diversity (~60%).
    if math.random() < 0.40 then
        return (math.random() < 0.5) and bl.bl_small or bl.bl_big
    end

    local candidates = {}
    local is_glass_mode = (archetype == 'all_glass_shatter' or archetype == 'partial_glass_shatter')
    for k, v in pairs(bl) do
        if type(v) == 'table' and k ~= 'bl_small' and k ~= 'bl_big' then
            -- Skip card-debuffing blinds in glass-shatter archetypes so Glass shatter isn't suppressed.
            local has_debuff = v.debuff and (v.debuff.suit or v.debuff.is_face or v.debuff.value or v.debuff.nominal)
            local suppresses_glass = is_glass_mode and (has_debuff or v.name == 'Verdant Leaf' or v.name == 'The Flint')
            if not suppresses_glass then
                candidates[#candidates + 1] = v
            end
        end
    end
    if #candidates > 0 then
        return candidates[math.random(#candidates)]
    end
    return bl.bl_small
end

--- Configure JokerDisplay mode for visual variety in jokers above play_area.
local function configure_joker_display_for_clip()
    if not (JokerDisplay and type(JokerDisplay.config) == 'table') then
        return false
    end
    local enabled = (math.random() < 0.65)
    JokerDisplay.config.enabled = enabled
    JokerDisplay.config.hide_by_default = false
    return enabled
end

--- Build 0..8 jokers tailored to the clip archetype and emplace them in G.jokers.
local function populate_jokers(mod, archetype, jd_enabled)
    local num_jokers
    if archetype == 'fast_clean' then
        num_jokers = math.random(0, 5)
    elseif archetype == 'heavy_retrigger' or archetype == 'end_round_jokers' then
        num_jokers = math.random(3, 8)
    else
        num_jokers = math.random(1, 7)
    end

    if not G.jokers or not G.jokers.config then return 0 end
    G.jokers.config.card_limit = math.max(5, num_jokers)

    local all_jokers = (mod.ShopGen and mod.ShopGen.get_all_jokers) and mod.ShopGen.get_all_jokers(true) or {}
    local speciality_pool = (archetype == 'heavy_retrigger' and RETRIGGER_JOKERS)
        or (archetype == 'end_round_jokers' and END_ROUND_JOKERS)
        or (archetype == 'held_end_round' and { 'j_mime', 'j_blueprint', 'j_brainstorm', 'j_baron', 'j_raised_fist' })
        or nil

    local placed = 0
    for i = 1, num_jokers do
        local center = nil
        if speciality_pool and (#speciality_pool > 0) and (i <= 3 or math.random() < 0.60) then
            local key = speciality_pool[math.random(#speciality_pool)]
            center = G.P_CENTERS and G.P_CENTERS[key] or nil
        end
        if not center and #all_jokers > 0 then
            center = all_jokers[math.random(#all_jokers)]
        end

        if center then
            local card = mod:create_random_joker_card(G.jokers, center, false)
            if card then
                -- Keep jokers front-facing unless Amber Acorn explicitly flips them later.
                card.facing = 'front'
                card.sprite_facing = 'front'
                -- Avoid triggering shop/sell-cost surprises while adding to deck.
                card.cost = math.random(2, 10)
                card:set_cost()
                pcall(function() card:add_to_deck() end)
                G.jokers:emplace(card)
                if jd_enabled and card.update_joker_display then
                    pcall(function() card:update_joker_display(true, true, "gdcv_end_round") end)
                elseif not jd_enabled and card.joker_display_remove then
                    pcall(function() card:joker_display_remove() end)
                end
                placed = placed + 1
            end
        end
    end

    snap_cardarea(G.jokers)
    return placed
end

--- Create a playing card with controlled enhancement/seal/edition probabilities.
local SUIT_PREFIXES = { 'H', 'C', 'D', 'S' }
local RANK_SUFFIXES = { '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A' }
local SAFE_KICKER_ENHANCEMENTS = { 'c_base', 'm_bonus', 'm_mult', 'm_lucky', 'm_gold', 'm_steel' }

--- Create a randomized playing card in `area` with optional enhancement, seal, flip state, and base center.
local function create_custom_playing_card(mod, area, force_enhancement, force_seal, stay_flipped, forced_base_card)
    local base_card = forced_base_card or mod:get_random_card_center()
    if not base_card then return nil end

    local center = (force_enhancement and G.P_CENTERS and G.P_CENTERS[force_enhancement]) or nil
    if not center then
        if math.random() < 0.55 then
            local enh_key = ENHANCEMENT_CHOICES[math.random(#ENHANCEMENT_CHOICES)]
            center = G.P_CENTERS and G.P_CENTERS[enh_key]
        end
        center = center or (G.P_CENTERS and G.P_CENTERS.c_base)
    end

    local card = Card(
        area.T.x + area.T.w * 0.5, area.T.y,
        G.CARD_W, G.CARD_H,
        base_card, center,
        {
            bypass_discovery_center = true,
            bypass_discovery_ui = true,
            bypass_back = G.GAME and G.GAME.selected_back and G.GAME.selected_back.pos,
        }
    )
    if card.set_base then card:set_base(base_card, true) end

    local ed = mod.get_random_edition and mod:get_random_edition() or nil
    if ed then card:set_edition({ [ed] = true }, true, true) end

    local seal = force_seal
    if seal == nil and math.random() < 0.30 then
        seal = SEAL_CHOICES[math.random(#SEAL_CHOICES)]
    end
    if seal and seal ~= false then
        card:set_seal(seal, true, true)
    end

    card.ability = card.ability or {}
    card.ability.set = (center and center.set) or 'Default'
    card:add_to_deck()
    table.insert(G.playing_cards, card)

    local flipped = stay_flipped
    if flipped == nil and G.GAME and G.GAME.blind then
        flipped = G.GAME.blind:stay_flipped(area, card)
    end
    area:emplace(card, nil, flipped)
    if G.GAME and G.GAME.blind then
        G.GAME.blind:debuff_card(card)
    end
    return card
end

--- Build explicit base cards for glass-shatter hands so the intended Glass cards always score.
local function build_played_specs(archetype, num_play)
    local specs = {}
    if archetype == 'all_glass_shatter' then
        if num_play == 5 and math.random() < 0.45 then
            -- 5-card Flush: same suit, distinct ranks -> all 5 cards score and shatter.
            local suit = SUIT_PREFIXES[math.random(#SUIT_PREFIXES)]
            local start_idx = math.random(1, #RANK_SUFFIXES - 4)
            for i = 1, 5 do
                specs[i] = {
                    base = find_p_card(suit, RANK_SUFFIXES[start_idx + i - 1]),
                    enh = 'm_glass',
                }
            end
        elseif num_play == 5 then
            -- Full House (3 + 2): standard 5-card hand where all 5 cards score and shatter.
            local r1_idx = math.random(#RANK_SUFFIXES)
            local r2_idx = ((r1_idx + math.random(#RANK_SUFFIXES - 1) - 1) % #RANK_SUFFIXES) + 1
            for i = 1, 5 do
                local rank = (i <= 3) and RANK_SUFFIXES[r1_idx] or RANK_SUFFIXES[r2_idx]
                local suit = SUIT_PREFIXES[((i - 1) % #SUIT_PREFIXES) + 1]
                specs[i] = { base = find_p_card(suit, rank), enh = 'm_glass' }
            end
        else
            -- 1..4 cards sharing one rank (High Card / Pair / 3oak / 4oak) -> all score and shatter.
            local rank = RANK_SUFFIXES[math.random(#RANK_SUFFIXES)]
            for i = 1, num_play do
                local suit = SUIT_PREFIXES[((i - 1) % #SUIT_PREFIXES) + 1]
                specs[i] = { base = find_p_card(suit, rank), enh = 'm_glass' }
            end
        end
        return specs
    end

    if archetype == 'partial_glass_shatter' then
        -- Guarantee the Glass subset forms the sole scoring combination while kickers stay non-scoring.
        local num_glass = math.random(1, math.max(1, num_play - 1))
        if num_glass > 3 then num_glass = 2 end
        local glass_rank = (num_glass == 1) and 'A' or 'K'
        -- Non-consecutive low/mid kicker ranks across alternating suits avoid accidental Pair/Straight/Flush.
        local kicker_ranks = { '2', '4', '7', '9', 'J' }
        local kicker_i = 1
        for i = 1, num_play do
            if i <= num_glass then
                local suit = SUIT_PREFIXES[((i - 1) % #SUIT_PREFIXES) + 1]
                specs[i] = { base = find_p_card(suit, glass_rank), enh = 'm_glass' }
            else
                local suit = SUIT_PREFIXES[((kicker_i) % #SUIT_PREFIXES) + 1]
                local rank = kicker_ranks[kicker_i] or '3'
                kicker_i = kicker_i + 1
                specs[i] = {
                    base = find_p_card(suit, rank),
                    enh = SAFE_KICKER_ENHANCEMENTS[math.random(#SAFE_KICKER_ENHANCEMENTS)],
                }
            end
        end
        return specs
    end

    for i = 1, num_play do
        local force_enh = nil
        local force_seal = nil
        if archetype == 'heavy_retrigger' then
            force_seal = (math.random() < 0.70) and 'Red' or nil
            force_enh = (math.random() < 0.70) and ({ 'm_lucky', 'm_mult', 'm_bonus', 'm_glass' })[math.random(4)] or nil
        end
        specs[i] = { enh = force_enh, seal = force_seal }
    end
    return specs
end

--- Populate G.hand, G.deck, and optionally G.consumeables and choose the played cards.
local function populate_cards_and_select(mod, archetype)
    local target_hand_size = math.random(6, 9)
    -- The Hook discards 2 cards from G.hand on press_play; ensure enough cards remain unhighlighted.
    if G.GAME and G.GAME.blind and G.GAME.blind.name == 'The Hook' then
        target_hand_size = math.max(target_hand_size, 7)
    end
    if G.hand and G.hand.config then
        G.hand.config.card_limit = math.max(8, target_hand_size)
        G.hand.config.highlighted_limit = 5
    end

    -- Select 1..5 cards to play (at least 3 for partial shatter so both shatter & slide occur; 5 for The Psychic).
    local min_play = (archetype == 'partial_glass_shatter') and 3 or 1
    local num_play = math.random(min_play, 5)
    if G.GAME and G.GAME.blind and G.GAME.blind.name == 'The Psychic' then
        num_play = 5
    end

    local played_specs = build_played_specs(archetype, num_play)
    local to_highlight = {}
    for i = 1, target_hand_size do
        local is_played = (i <= num_play)
        local force_enh = nil
        local force_seal = nil
        local forced_base = nil

        if is_played then
            local spec = played_specs[i] or {}
            force_enh = spec.enh
            force_seal = spec.seal
            forced_base = spec.base
        else
            -- Held cards remaining in G.hand after play (important for end-of-round held effects + discard sweep).
            if archetype == 'held_end_round' then
                force_enh = (math.random() < 0.75) and 'm_gold' or 'm_steel'
                force_seal = (math.random() < 0.50) and ({ 'Blue', 'Red' })[math.random(2)] or nil
            end
        end

        local card = create_custom_playing_card(mod, G.hand, force_enh, force_seal, nil, forced_base)
        if card then
            if is_played then
                -- Prevent Boss debuffs on scoring glass cards so shatter triggers reliably.
                if archetype == 'all_glass_shatter' or (archetype == 'partial_glass_shatter' and force_enh == 'm_glass') then
                    card:set_debuff(false)
                end
                to_highlight[#to_highlight + 1] = card
            end
        end
    end

    -- Shuffle highlight order slightly so played cards are spread across G.hand visually.
    G.hand:sort()
    snap_cardarea(G.hand)

    G.hand:unhighlight_all()
    for _, card in ipairs(to_highlight) do
        G.hand:add_to_highlighted(card, true)
    end
    snap_cardarea(G.hand)

    -- Populate deck so remaining deck count and shadows look realistic.
    local deck_count = math.random(12, 34)
    for _ = 1, deck_count do
        create_custom_playing_card(mod, G.deck, nil, nil, true)
    end
    snap_cardarea(G.deck)

    -- Populate 0..2 consumables so Blue Seals still have room (~70% leave space) and top-right looks natural.
    if G.consumeables and G.consumeables.config then
        G.consumeables.config.card_limit = math.max(2, G.consumeables.config.card_limit or 2)
        local n_cons = math.random(0, G.consumeables.config.card_limit - (archetype == 'held_end_round' and 1 or 0))
        local c_types = { 'Tarot', 'Planet', 'Spectral' }
        for _ = 1, n_cons do
            local c = mod:create_random_consumable_card(G.consumeables, c_types[math.random(3)])
            if c then G.consumeables:emplace(c) end
        end
        snap_cardarea(G.consumeables)
    end

    return #to_highlight
end

--- Apply blind, chips, round money, and guarantee probabilities before cards are dealt.
local function configure_blind_and_round(blind_center, archetype)
    G.GAME.round_resets = G.GAME.round_resets or {}
    G.GAME.round_resets.ante = math.random(1, 8)
    G.GAME.win_ante = 999
    G.GAME.round = math.random(1, 24)
    G.GAME.dollars = math.random(0, 85)

    G.GAME.current_round.hands_left = math.random(1, 5)
    G.GAME.current_round.discards_left = math.random(0, 4)

    if G.GAME.blind and blind_center then
        G.GAME.blind:set_blind(blind_center, false, true)
        -- Disable Crimson Heart / Verdant Leaf / Amber Acorn only when they interfere with glass shatter.
        if archetype == 'all_glass_shatter' then
            G.GAME.blind.disabled = true
        end
        G.GAME.blind.chips = 1
        G.GAME.blind.chip_text = number_format(1)
        G.GAME.blind.block_play = nil
        G.GAME.blind.blind_set = true
        if G.GAME.blind.children and G.GAME.blind.children.animatedSprite and blind_center.pos then
            G.GAME.blind.children.animatedSprite:set_sprite_pos(blind_center.pos)
        end
        if G.HUD_blind then
            G.HUD_blind.alignment.offset.y = 0
            local name_uie = G.HUD_blind:get_UIE_by_ID("HUD_blind_name")
            if name_uie then
                if name_uie.states then name_uie.states.visible = true end
                if name_uie.config and name_uie.config.object and name_uie.config.object.pop_in then
                    name_uie.config.object:pop_in(0)
                end
            end
            local d_uie = G.HUD_blind:get_UIE_by_ID("dollars_to_be_earned")
            if d_uie then
                if d_uie.parent and d_uie.parent.parent and d_uie.parent.parent.states then
                    d_uie.parent.parent.states.visible = true
                end
                if d_uie.config and d_uie.config.object and d_uie.config.object.pop_in then
                    d_uie.config.object:pop_in(0)
                end
            end
            G.HUD_blind:recalculate(false)
            if G.HUD_blind.hard_set_VT then G.HUD_blind:hard_set_VT() end
            if G.HUD and G.HUD.recalculate then G.HUD:recalculate() end
        end
    end

    -- Preserve Oops! All 6s multipliers already applied by add_to_deck(), while guaranteeing 100% Glass shatter in all_glass_shatter.
    local cur_prob = (G.GAME.probabilities and G.GAME.probabilities.normal) or 1
    if archetype == 'all_glass_shatter' then
        G.GAME.probabilities.normal = math.max(4, cur_prob)
    else
        G.GAME.probabilities.normal = math.max(cur_prob, (math.random() < 0.15) and 2 or 1)
    end
end

--- Ensure a single played hand clears the blind (even if Boss debuffs the hand), and reset hand text.
local function finalize_chips_and_hand_text()
    local debuffed_hand = false
    if G.hand and #G.hand.highlighted > 0 and G.FUNCS and G.FUNCS.get_poker_hand_info and G.GAME.blind then
        local _, _, ph = G.FUNCS.get_poker_hand_info(G.hand.highlighted)
        debuffed_hand = G.GAME.blind:debuff_hand(G.hand.highlighted, ph, nil, true) == true
    end

    G.GAME.chips = (debuffed_hand or math.random() < 0.05) and 1 or 0
    if update_hand_text then
        update_hand_text({ immediate = true, nopulse = true, delay = 0 }, {
            handname = '', level = '', mult = 0, chips = 0,
        })
    end
end

--- Prepare a complete randomized board state ready to play immediately on clip frame 0.
--- Returns a scenario metadata table to store in `meta.json`.
function EndRoundGen.setup_clip_scenario(mod)
    EndRoundGen.backup_user_state(mod)
    EndRoundGen.cleanup_clip_state(mod)

    local archetype = weighted_choice(ARCHETYPE_WEIGHTS)
    local gamespeed = weighted_choice(GAMESPEED_WEIGHTS)
    local crt = weighted_choice(CRT_WEIGHTS)
    local bloom = (math.random() < 0.72) and 1 or 2
    local reduced_motion = (math.random() < 0.18)

    G.SETTINGS.GAMESPEED = gamespeed
    G.SETTINGS.reduced_motion = reduced_motion
    if G.SETTINGS.GRAPHICS then
        G.SETTINGS.GRAPHICS.crt = crt
        G.SETTINGS.GRAPHICS.bloom = bloom
    end

    local simulate_hover = (math.random() < 0.14)

    -- Visual diversity: shaders, deck backs/skins, high-contrast cards, JokerDisplay overlay toggle.
    if mod.randomize_background then mod:randomize_background() end
    if mod.randomize_deck_appearance then mod:randomize_deck_appearance() end
    local jd_enabled = configure_joker_display_for_clip()

    local blind_center = pick_blind_for_archetype(archetype)
    local num_jokers = populate_jokers(mod, archetype, jd_enabled)
    configure_blind_and_round(blind_center, archetype)
    local num_played = populate_cards_and_select(mod, archetype)
    finalize_chips_and_hand_text()

    -- Discard any events queued cleanly by card:add_to_deck / blind setup before frame 0 starts.
    EndRoundGen.sweep_transient_effects(mod)
    if G.GAME and G.GAME.blind then
        G.GAME.blind.block_play = nil
        G.GAME.blind:set_text()
    end

    local hover_target = nil
    if simulate_hover and G.jokers and #G.jokers.cards > 0 then
        hover_target = G.jokers.cards[math.random(#G.jokers.cards)]
    end
    G.STATE = G.STATES.SELECTING_HAND
    G.STATE_COMPLETE = true

    return {
        archetype = archetype,
        gamespeed = gamespeed,
        crt = crt,
        bloom = bloom,
        reduced_motion = reduced_motion,
        joker_display_enabled = jd_enabled,
        simulate_hover = (hover_target ~= nil),
        hover_target = hover_target,
        blind_key = blind_center and blind_center.key or 'bl_small',
        blind_name = blind_center and blind_center.name or 'Small Blind',
        num_jokers = num_jokers,
        num_played_cards = num_played,
        num_held_cards = (G.hand and G.hand.cards) and (#G.hand.cards - num_played) or 0,
    }
end

return EndRoundGen
