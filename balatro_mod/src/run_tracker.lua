-- gamestate_dataset_cv/src/run_tracker.lua
-- Dedicated telemetry logger tracking exact ground truth gamestates at each action frame.

local RunTracker = {}

RunTracker.enabled = false
RunTracker.log_file = nil
RunTracker.action_step = 0

function RunTracker.init()
    RunTracker.enabled = true
    RunTracker.action_step = 0
    local log_path = "run_telemetry.jsonl"
    RunTracker.log_file = log_path
    love.filesystem.write(log_path, "")
    print("RUN_TRACKER: Initialized telemetry logger -> " .. tostring(log_path))
end

local function serialize_card(c)
    if not c then return nil end
    local edition_name = "none"
    if c.edition then
        if c.edition.foil then edition_name = "foil"
        elseif c.edition.holo then edition_name = "holographic"
        elseif c.edition.polychrome then edition_name = "polychrome"
        elseif c.edition.negative then edition_name = "negative"
        end
    end

    local seal_name = "none"
    if c.seal then
        seal_name = string.lower(c.seal)
    end

    local enhancement_name = "none"
    if c.ability and c.ability.name then
        local en = c.ability.name
        if en:find("Bonus") then enhancement_name = "bonus"
        elseif en:find("Mult") then enhancement_name = "mult"
        elseif en:find("Wild") then enhancement_name = "wild"
        elseif en:find("Glass") then enhancement_name = "glass"
        elseif en:find("Steel") then enhancement_name = "steel"
        elseif en:find("Stone") then enhancement_name = "stone"
        elseif en:find("Gold") then enhancement_name = "gold"
        elseif en:find("Lucky") then enhancement_name = "lucky"
        end
    end

    return {
        key = c.config and c.config.center and c.config.center.key or nil,
        name = c.ability and c.ability.name or (c.base and c.base.name) or "Unknown",
        cost = c.cost or 0,
        sell_cost = c.sell_cost or 0,
        extra_value = c.ability and c.ability.extra_value or 0,
        edition = edition_name,
        seal = seal_name,
        enhancement = enhancement_name,
        debuffed = c.debuff or false,
        suit = c.base and c.base.suit or "none",
        value = c.base and c.base.value or "none"
    }
end

function RunTracker.snapshot_gamestate(action_name, action_metadata)
    if not RunTracker.enabled or not G.GAME then return end

    RunTracker.action_step = RunTracker.action_step + 1

    local jokers = {}
    if G.jokers and G.jokers.cards then
        for i, j in ipairs(G.jokers.cards) do
            local item = serialize_card(j)
            item.mult = j.ability and j.ability.mult or 0
            item.x_mult = j.ability and j.ability.x_mult or 1.0
            item.chips = j.ability and j.ability.chips or 0
            item.h_size = j.ability and j.ability.h_size or 0
            table.insert(jokers, item)
        end
    end

    local consumables = {}
    if G.consumeables and G.consumeables.cards then
        for i, c in ipairs(G.consumeables.cards) do
            table.insert(consumables, serialize_card(c))
        end
    end

    local hand_cards = {}
    local selected_cards = {}
    if G.hand and G.hand.cards then
        for i, c in ipairs(G.hand.cards) do
            local sc = serialize_card(c)
            sc.selected = c.highlighted or false
            table.insert(hand_cards, sc)
            if c.highlighted then
                table.insert(selected_cards, sc)
            end
        end
    end

    local shop_cards = {}
    if G.shop_jokers and G.shop_jokers.cards then
        for i, c in ipairs(G.shop_jokers.cards) do
            table.insert(shop_cards, serialize_card(c))
        end
    end

    local vouchers_redeemed = {}
    if G.GAME.used_vouchers then
        for k, v in pairs(G.GAME.used_vouchers) do
            if v then table.insert(vouchers_redeemed, k) end
        end
    end

    local blind_name = G.GAME.blind and G.GAME.blind.name or "Small Blind"
    local boss_name = G.GAME.round_resets and G.GAME.round_resets.blind_choices and G.GAME.round_resets.blind_choices.Boss or nil

    local full_deck_cards = {}
    if G.playing_cards then
        for i, c in ipairs(G.playing_cards) do
            table.insert(full_deck_cards, serialize_card(c))
        end
    end

    local discard_cards = {}
    if G.discard and G.discard.cards then
        for i, c in ipairs(G.discard.cards) do
            table.insert(discard_cards, serialize_card(c))
        end
    end

    local payload = {
        step = RunTracker.action_step,
        timestamp = love.timer.getTime(),
        action = action_name,
        action_metadata = action_metadata or {},
        state = {
            money = G.GAME.dollars or 0,
            ante = G.GAME.round_resets and G.GAME.round_resets.ante or 1,
            round = G.GAME.round or 1,
            hands_left = G.GAME.current_round and G.GAME.current_round.hands_left or 0,
            discards_left = G.GAME.current_round and G.GAME.current_round.discards_left or 0,
            hands_played_this_round = G.GAME.current_round and G.GAME.current_round.hands_played or 0,
            discards_used_this_round = G.GAME.current_round and G.GAME.current_round.discards_used or 0,
            reroll_cost = G.GAME.current_round and G.GAME.current_round.reroll_cost or 5,
            target_chips = G.GAME.blind and G.GAME.blind.chips or 0,
            current_chips = G.GAME.chips or 0,
            blind = blind_name,
            boss_blind = boss_name,
            jokers = jokers,
            consumables = consumables,
            hand = hand_cards,
            selected_cards = selected_cards,
            shop = shop_cards,
            vouchers = vouchers_redeemed,
            deck_cards_count = (G.playing_cards and #G.playing_cards) or 52,
            deck = full_deck_cards,
            discard_pile = discard_cards,
            remaining_deck_count = (G.deck and G.deck.cards and #G.deck.cards) or 0
        }
    }

    local json_str = (JSON and JSON.encode) and JSON.encode(payload) or nil
    if not json_str and G.JSON and G.JSON.encode then
        json_str = G.JSON.encode(payload)
    end

    if json_str then
        love.filesystem.append(RunTracker.log_file, json_str .. "\n")
    end
end

-- ========================================================
-- ACTION HOOKS (Intercept exact actions matching templates)
-- ========================================================

function RunTracker.hook_actions()
    -- 1. Play Hand
    local orig_play = G.FUNCS.play_cards_from_highlighted
    G.FUNCS.play_cards_from_highlighted = function(e)
        RunTracker.snapshot_gamestate("Play Hand")
        return orig_play(e)
    end

    -- 2. Discard
    local orig_discard = G.FUNCS.discard_cards_from_highlighted
    G.FUNCS.discard_cards_from_highlighted = function(e, hook)
        RunTracker.snapshot_gamestate("Discard")
        return orig_discard(e, hook)
    end

    -- 3. Reroll Shop
    local orig_reroll = G.FUNCS.reroll_shop
    G.FUNCS.reroll_shop = function(e)
        RunTracker.snapshot_gamestate("Reroll Shop")
        return orig_reroll(e)
    end

    -- 4. Cash Out
    local orig_cash_out = G.FUNCS.cash_out
    G.FUNCS.cash_out = function(e)
        RunTracker.snapshot_gamestate("Cash Out")
        return orig_cash_out(e)
    end

    -- 5. Buy Card
    local orig_buy = G.FUNCS.buy_from_shop
    G.FUNCS.buy_from_shop = function(e)
        local card = e and e.config and e.config.ref_table
        local act_name = (e and e.config and e.config.id == 'buy_and_use') and "Buy and Use" or "Buy Card"
        RunTracker.snapshot_gamestate(act_name, {card = card and card.ability and card.ability.name or nil})
        return orig_buy(e)
    end

    -- 6. Sell Card
    local orig_sell = G.FUNCS.sell_card
    G.FUNCS.sell_card = function(e)
        local card = e and e.config and e.config.ref_table
        RunTracker.snapshot_gamestate("Sell Card", {card = card and card.ability and card.ability.name or nil})
        return orig_sell(e)
    end

    -- 7. Select Blind
    local orig_select_blind = G.FUNCS.select_blind
    if orig_select_blind then
        G.FUNCS.select_blind = function(e)
            RunTracker.snapshot_gamestate("Select Blind")
            return orig_select_blind(e)
        end
    end

    -- 8. Skip Blind
    local orig_skip_blind = G.FUNCS.skip_blind
    if orig_skip_blind then
        G.FUNCS.skip_blind = function(e)
            RunTracker.snapshot_gamestate("Skip Blind")
            return orig_skip_blind(e)
        end
    end

    print("RUN_TRACKER: All gameplay action hooks installed!")
end

return RunTracker
