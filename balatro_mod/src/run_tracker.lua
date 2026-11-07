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

    -- Mirror to workspace repo root if NFS is available
    if NFS and NFS.write and G.DATASET_MOD and G.DATASET_MOD.mod_path then
        local repo_log = G.DATASET_MOD.mod_path .. "../run_telemetry.jsonl"
        NFS.write(repo_log, "")
        RunTracker.repo_log_file = repo_log
        print("RUN_TRACKER: Initialized repo mirror -> " .. tostring(repo_log))
    end
end

local function find_card_index_0based(area, target)
    if not area or not area.cards or not target then return nil end
    for i, c in ipairs(area.cards) do
        if c == target then
            return i - 1
        end
    end
    return nil
end

local function sanitize_extra(val)
    local t = type(val)
    if t == "number" or t == "string" or t == "boolean" then
        return val
    elseif t == "table" then
        local out = {}
        for k, v in pairs(val) do
            if type(k) == "string" or type(k) == "number" then
                local sv = sanitize_extra(v)
                if sv ~= nil then
                    out[tostring(k)] = sv
                end
            end
        end
        return out
    end
    return nil
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
        id = c.sort_id or 0,
        key = c.config and c.config.center and c.config.center.key or nil,
        set = c.ability and c.ability.set or nil,
        name = c.ability and c.ability.name or (c.base and c.base.name) or "Unknown",
        cost = c.cost or 0,
        sell_cost = c.sell_cost or 0,
        extra_value = c.ability and c.ability.extra_value or 0,
        bonus_chips = c.ability and c.ability.perma_bonus or 0,
        edition = edition_name,
        seal = seal_name,
        enhancement = enhancement_name,
        debuffed = c.debuff or false,
        eternal = (c.ability and c.ability.eternal) or false,
        perishable = (c.ability and c.ability.perishable) or false,
        perish_tally = (c.ability and c.ability.perish_tally) or 5,
        rental = (c.ability and c.ability.rental) or false,
        facing_back = (c.facing == "back") or false,
        forced_selection = (c.ability and c.ability.forced_selection) or false,
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
            local ab = j.ability or {}
            local ex = type(ab.extra) == "table" and ab.extra or nil
            item.mult = ab.mult or (ex and ex.mult) or 0
            item.x_mult = ab.x_mult or (ex and ex.x_mult) or 1.0
            item.chips = (ex and ex.chips) or ab.chips or (type(ab.extra) == "number" and ab.extra) or 0
            item.h_size = (ex and ex.h_size) or ab.h_size or 0
            item.invis_rounds = ab.invis_rounds or 0
            item.yorick_discards = ab.yorick_discards or nil
            item.to_do_poker_hand = ab.to_do_poker_hand or nil
            item.extra = sanitize_extra(ab.extra)
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

    local shop_vouchers = {}
    if G.shop_vouchers and G.shop_vouchers.cards then
        for i, c in ipairs(G.shop_vouchers.cards) do
            table.insert(shop_vouchers, serialize_card(c))
        end
    end

    local shop_packs = {}
    if G.shop_booster and G.shop_booster.cards then
        for i, c in ipairs(G.shop_booster.cards) do
            local sp = serialize_card(c)
            sp.choose = (c.config and c.config.center and c.config.center.config and c.config.center.config.choose) or 1
            sp.extra = (c.ability and c.ability.extra) or 3
            table.insert(shop_packs, sp)
        end
    end

    local pack_cards = {}
    if G.pack_cards and G.pack_cards.cards then
        for i, c in ipairs(G.pack_cards.cards) do
            table.insert(pack_cards, serialize_card(c))
        end
    end

    local vouchers_redeemed = {}
    if G.GAME.used_vouchers then
        for k, v in pairs(G.GAME.used_vouchers) do
            if v then table.insert(vouchers_redeemed, k) end
        end
    end

    local hand_levels = {}
    local hand_play_counts = {}
    if G.GAME.hands then
        for k, v in pairs(G.GAME.hands) do
            hand_levels[k] = v.level or 1
            hand_play_counts[k] = v.played or 0
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

    local cr = G.GAME.current_round or {}

    local payload = {
        step = RunTracker.action_step,
        timestamp = love.timer.getTime(),
        action = action_name,
        action_metadata = action_metadata or {},
        state = {
            money = G.GAME.dollars or 0,
            round_dollars = cr.dollars or 0,
            ante = G.GAME.round_resets and G.GAME.round_resets.ante or 1,
            round = G.GAME.round or 1,
            hands_left = cr.hands_left or 0,
            discards_left = cr.discards_left or 0,
            hands_played_this_round = cr.hands_played or 0,
            discards_used_this_round = cr.discards_used or 0,
            reroll_cost = cr.reroll_cost or 5,
            free_rerolls = cr.free_rerolls or 0,
            blinds_skipped = G.GAME.skips or 0,
            tarots_used = (G.GAME.consumeable_usage_total and G.GAME.consumeable_usage_total.tarot) or 0,
            unused_discards = G.GAME.unused_discards or 0,
            last_tarot_planet = G.GAME.last_tarot_planet or nil,
            castle_suit = (cr.castle_card and cr.castle_card.suit) or "Spades",
            ancient_suit = (cr.ancient_card and cr.ancient_card.suit) or "Spades",
            mail_card_rank = (cr.mail_card and cr.mail_card.value) or "Ace",
            idol_card = cr.idol_card and {rank = cr.idol_card.value, suit = cr.idol_card.suit} or nil,
            target_chips = G.GAME.blind and G.GAME.blind.chips or 0,
            current_chips = G.GAME.chips or 0,
            blind = blind_name,
            boss_blind = boss_name,
            jokers = jokers,
            joker_slots = (G.jokers and G.jokers.config and G.jokers.config.card_limit) or 5,
            consumables = consumables,
            consumable_slots = (G.consumeables and G.consumeables.config and G.consumeables.config.card_limit) or 2,
            hand = hand_cards,
            hand_max_size = (G.hand and G.hand.config and G.hand.config.card_limit) or 8,
            selected_cards = selected_cards,
            shop = shop_cards,
            shop_vouchers = shop_vouchers,
            shop_packs = shop_packs,
            pack_cards = pack_cards,
            vouchers = vouchers_redeemed,
            hand_levels = hand_levels,
            hand_play_counts = hand_play_counts,
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
        if RunTracker.repo_log_file and NFS and NFS.append then
            NFS.append(RunTracker.repo_log_file, json_str .. "\n")
        end
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
        RunTracker.snapshot_gamestate("Cash Out", {
            include_round_payout = true,
            round_dollars = G.GAME and G.GAME.current_round and G.GAME.current_round.dollars or nil
        })
        return orig_cash_out(e)
    end

    -- 5. Buy Card / Buy and Use
    local orig_buy = G.FUNCS.buy_from_shop
    G.FUNCS.buy_from_shop = function(e)
        local card = e and e.config and e.config.ref_table
        local act_name = (e and e.config and e.config.id == 'buy_and_use') and "Buy and Use" or "Buy Card"
        local idx = find_card_index_0based(G.shop_jokers, card)
        RunTracker.snapshot_gamestate(act_name, {
            card = card and card.ability and card.ability.name or nil,
            key = card and card.config and card.config.center and card.config.center.key or nil,
            cost = card and card.cost or 0,
            target_idx = idx
        })
        return orig_buy(e)
    end

    -- 6. Sell Card
    local orig_sell = G.FUNCS.sell_card
    G.FUNCS.sell_card = function(e)
        local card = e and e.config and e.config.ref_table
        local idx = find_card_index_0based(G.jokers, card)
        if idx == nil and G.consumeables then
            local c_idx = find_card_index_0based(G.consumeables, card)
            if c_idx ~= nil then
                idx = ((G.jokers and G.jokers.cards and #G.jokers.cards) or 0) + c_idx
            end
        end
        RunTracker.snapshot_gamestate("Sell Card", {
            card = card and card.ability and card.ability.name or nil,
            key = card and card.config and card.config.center and card.config.center.key or nil,
            sell_cost = card and card.sell_cost or 0,
            target_idx = idx
        })
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

    -- 9. Use Card (Use Consumable, Open Pack, Redeem Voucher, Use In Pack)
    local orig_use_card = G.FUNCS.use_card
    if orig_use_card then
        G.FUNCS.use_card = function(e, mute, nosave)
            if not (e and e.config and e.config.id == 'buy_and_use') then
                local card = e and e.config and e.config.ref_table
                if card and not (card.check_use and card:check_use()) then
                    local set_name = card.ability and card.ability.set or ""
                    local act_name = "Use Consumable"
                    local idx = nil
                    if set_name == "Booster" then
                        act_name = "Open Pack"
                        idx = find_card_index_0based(G.shop_booster, card)
                    elseif set_name == "Voucher" then
                        act_name = "Redeem Voucher"
                        idx = find_card_index_0based(G.shop_vouchers, card)
                    elseif card.area == G.pack_cards then
                        act_name = "Use In Pack"
                        idx = find_card_index_0based(G.pack_cards, card)
                    else
                        act_name = "Use Consumable"
                        idx = find_card_index_0based(G.consumeables, card)
                    end
                    RunTracker.snapshot_gamestate(act_name, {
                        card = card.ability and card.ability.name or nil,
                        key = card.config and card.config.center and card.config.center.key or nil,
                        set = set_name,
                        cost = card.cost or 0,
                        target_idx = idx
                    })
                end
            end
            return orig_use_card(e, mute, nosave)
        end
    end

    -- 10. Skip Booster Pack
    local orig_skip_booster = G.FUNCS.skip_booster
    if orig_skip_booster then
        G.FUNCS.skip_booster = function(e)
            RunTracker.snapshot_gamestate("Skip Pack")
            return orig_skip_booster(e)
        end
    end

    -- 11. Next Round (Leave Shop)
    local orig_toggle_shop = G.FUNCS.toggle_shop
    if orig_toggle_shop then
        G.FUNCS.toggle_shop = function(e)
            RunTracker.snapshot_gamestate("Next Round")
            return orig_toggle_shop(e)
        end
    end

    -- 12. Reroll Boss Blind
    local orig_reroll_boss = G.FUNCS.reroll_boss
    if orig_reroll_boss then
        G.FUNCS.reroll_boss = function(e)
            RunTracker.snapshot_gamestate("Reroll Boss")
            return orig_reroll_boss(e)
        end
    end

    print("RUN_TRACKER: All 12 gameplay action hooks installed!")
end

return RunTracker
