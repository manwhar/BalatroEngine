import copy

from .context import TriggerContext, TriggerEvent
from .jokers import (
    dispatch_jokers,
    get_card_repetitions,
    get_oops_multiplier,
    on_joker_added,
    on_joker_removed,
    resolve_copy_target,
    sync_dynamic_joker_states,
)
from .poker_hands import evaluate_poker_hand, get_scoring_cards

try:
    from game.engine.rng_sim.events import (
        evaluate_lucky_hit,
        poll_edition,
        predict_familiar_spawns,
        predict_grim_spawns,
        predict_immolate_destructions,
        predict_incantation_spawns,
        predict_purple_seal_tarot,
        predict_random_destroy,
    )
except ImportError:
    from engine.rng_sim.events import (
        evaluate_lucky_hit,
        poll_edition,
        predict_familiar_spawns,
        predict_grim_spawns,
        predict_immolate_destructions,
        predict_incantation_spawns,
        predict_purple_seal_tarot,
        predict_random_destroy,
    )

try:
    from game.balatro_models import (
        Action,
        ActionType,
        BlindName,
        CardType,
        Deck,
        Edition,
        Enhancement,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        SpectralName,
        Stake,
        Suit,
        TarotName,
        Voucher,
        VoucherName,
    )
except ImportError:
    from balatro_models import (
        Action,
        ActionType,
        BlindName,
        CardType,
        Deck,
        Edition,
        Enhancement,
        JokerName,
        Location,
        Planet,
        PlanetName,
        PlayingCard,
        PokerHand,
        Rank,
        Seal,
        SkipTag,
        SpectralName,
        Stake,
        Suit,
        TarotName,
        Voucher,
        VoucherName,
    )

POKER_HAND_TO_PLANET = {
    PokerHand.PAIR: PlanetName.MERCURY,
    PokerHand.THREE_OF_A_KIND: PlanetName.VENUS,
    PokerHand.TWO_PAIR: PlanetName.URANUS,
    PokerHand.FULL_HOUSE: PlanetName.EARTH,
    PokerHand.FLUSH: PlanetName.JUPITER,
    PokerHand.STRAIGHT: PlanetName.SATURN,
    PokerHand.FOUR_OF_A_KIND: PlanetName.MARS,
    PokerHand.STRAIGHT_FLUSH: PlanetName.NEPTUNE,
    PokerHand.FIVE_OF_A_KIND: PlanetName.PLANET_X,
    PokerHand.FLUSH_HOUSE: PlanetName.CERES,
    PokerHand.FLUSH_FIVE: PlanetName.ERIS,
    PokerHand.HIGH_CARD: PlanetName.PLUTO,
}

SHOWDOWN_BOSSES = {
    BlindName.AMBER_ACORN,
    BlindName.VERDANT_LEAF,
    BlindName.VIOLET_VESSEL,
    BlindName.CRIMSON_HEART,
    BlindName.CERULEAN_BELL,
}


def calculate_base_round_payout(state) -> int:
    """Computes deterministic end-of-round base blind reward + remaining hands bonus + interest."""
    dollars = 0

    # 1. Base blind reward
    if state.current_blind == BlindName.SMALL:
        if state.stake == Stake.WHITE:
            dollars += 3
    elif state.current_blind == BlindName.BIG:
        dollars += 4
    elif state.current_blind in SHOWDOWN_BOSSES:
        dollars += 8
    else:
        dollars += 5

    # 2. Remaining hands & discards bonus
    if state.deck == Deck.GREEN_DECK:
        dollars += state.hands_left * 2 + state.discards_left * 1
    else:
        dollars += state.hands_left * 1

    # 3. Interest (calculated on state.money after rental deductions, before adding round payout)
    if state.deck != Deck.GREEN_DECK and state.money >= 5:
        if VoucherName.MONEY_TREE in state.vouchers_redeemed:
            interest_cap = 100
        elif VoucherName.SEED_MONEY in state.vouchers_redeemed:
            interest_cap = 50
        else:
            interest_cap = 25

        to_the_moon_count = sum(
            1
            for j in state.inventory.jokers
            if j.name == JokerName.TO_THE_MOON and not j.debuffed
        )
        interest_amount = 1 + to_the_moon_count
        dollars += interest_amount * min(state.money // 5, interest_cap // 5)

    return dollars


def trigger_end_of_round_effects(state, action: Action | None = None) -> None:
    """Executes Balatro's end_round() effects when a blind is beaten (or on Cash Out if not yet triggered)."""
    if getattr(state, "round_won", False):
        return
    state.round_won = True

    # 1. Held-in-hand end-of-round card effects (Gold cards + Red Seal + Mime retriggers)
    jokers = state.inventory.jokers if state.inventory else []
    mime_count = 0
    for j in jokers:
        if j.debuffed:
            continue
        if j.name == JokerName.MIME:
            mime_count += 1
        elif j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, jokers)
            if target and not target.debuffed and target.name == JokerName.MIME:
                mime_count += 1

    for hc in state.hand.cards:
        if hc.selected:
            continue
        c = hc.card
        if getattr(c, "debuffed", False):
            continue
        reps = 1 + (1 if c.seal == Seal.RED else 0) + mime_count
        if c.enhancement == Enhancement.GOLD:
            state.money += 3 * reps
        if c.seal == Seal.BLUE and state.inventory:
            for _ in range(reps):
                if state.inventory.can_add_consumable:
                    p_name = POKER_HAND_TO_PLANET.get(
                        getattr(state, "last_hand_played", PokerHand.HIGH_CARD),
                        PlanetName.PLUTO,
                    )
                    state.inventory.consumables.append(Planet(name=p_name))

    # 2. Unused discards, rental, perishable, and round-end Joker effects
    state.unused_discards += state.discards_left
    state.free_rerolls = 0
    for j in jokers:
        if j.rental:
            state.money -= 3
        if j.perishable:
            j.perish_tally = max(0, j.perish_tally - 1)
            if j.perish_tally == 0:
                j.debuffed = True

    ctx = TriggerContext(
        event=TriggerEvent.END_OF_ROUND,
        state=state,
        action=action,
        extra={"phase": "end_of_round"},
    )
    dispatch_jokers(ctx)
    state.discard_pile.clear()


def _evaluate_played_cards(state, action: Action, played_cards: list) -> None:
    """Runs the scoring pipeline for a played hand and triggers end_round() if the blind is beaten."""
    if state.current_blind == BlindName.THE_TOOTH and not state.boss_disabled:
        state.money -= len(played_cards)

    jokers = state.inventory.jokers if state.inventory else None
    scoring_cards = get_scoring_cards(played_cards, jokers)
    poker_hand = evaluate_poker_hand(played_cards, jokers)

    state.last_hand_played = poker_hand
    state.hand_play_counts[poker_hand] = state.hand_play_counts.get(poker_hand, 0) + 1
    state.hands_played_this_round_by_type.add(poker_hand)
    if (
        state.current_blind == BlindName.THE_MOUTH
        and not state.boss_disabled
        and state.locked_hand_type is None
    ):
        state.locked_hand_type = poker_hand

    # 1. Before play
    ctx = TriggerContext(
        event=TriggerEvent.BEFORE_PLAY,
        state=state,
        action=action,
        scoring_cards=scoring_cards,
        full_hand=played_cards,
        poker_hand=poker_hand,
    )
    dispatch_jokers(ctx)

    # 2. Individual card scored (for each active, non-debuffed scoring card + retriggers)
    for idx, card in enumerate(scoring_cards):
        if getattr(card, "debuffed", False):
            continue
        reps = get_card_repetitions(card, idx, state)
        for _ in range(reps):
            if getattr(card, "destroyed", False):
                break
            if card.seal == Seal.GOLD:
                state.money += 3
            extra = {}
            if card.enhancement == Enhancement.LUCKY and state.rng:
                hit = evaluate_lucky_hit(
                    state.rng, probability_modifier=get_oops_multiplier(state)
                )
                if hit.mult:
                    card.bonus_mult = getattr(card, "bonus_mult", 0) + 20
                if hit.money:
                    state.money += 20
                extra["lucky_triggered"] = hit.any_hit

            ctx_ind = TriggerContext(
                event=TriggerEvent.INDIVIDUAL_CARD_SCORED,
                state=state,
                action=action,
                target_card=card,
                scoring_cards=scoring_cards,
                full_hand=played_cards,
                poker_hand=poker_hand,
                extra=extra,
            )
            dispatch_jokers(ctx_ind)

    # 3. Held-in-hand individual card effects (Reserved Parking, Baron, Shoot the Moon retriggers)
    active_jokers = state.inventory.jokers if state.inventory else []
    mime_count = 0
    for j in active_jokers:
        if j.debuffed:
            continue
        if j.name == JokerName.MIME:
            mime_count += 1
        elif j.name in (JokerName.BLUEPRINT, JokerName.BRAINSTORM):
            target = resolve_copy_target(j, active_jokers)
            if target and not target.debuffed and target.name == JokerName.MIME:
                mime_count += 1

    held_cards = [hc.card for hc in state.hand.cards if not hc.selected]
    for card in held_cards:
        if getattr(card, "debuffed", False):
            continue
        reps = 1 + (1 if card.seal == Seal.RED else 0) + mime_count
        for _ in range(reps):
            ctx_held = TriggerContext(
                event=TriggerEvent.HELD_CARD_SCORED,
                state=state,
                action=action,
                target_card=card,
                scoring_cards=scoring_cards,
                full_hand=played_cards,
                poker_hand=poker_hand,
            )
            dispatch_jokers(ctx_held)

    # 4. Glass cards shatter check (evaluated once per scoring card after repetitions)
    for card in scoring_cards:
        if getattr(card, "debuffed", False) or getattr(card, "destroyed", False):
            continue
        if card.enhancement == Enhancement.GLASS and state.rng:
            if state.rng.poll_odds("glass", 1, 4, get_oops_multiplier(state)):
                card.destroyed = True
                state.remove_card_from_deck(card, shattered=True)

    # 5. After play
    ctx_after = TriggerContext(
        event=TriggerEvent.AFTER_PLAY,
        state=state,
        action=action,
        scoring_cards=scoring_cards,
        full_hand=played_cards,
        poker_hand=poker_hand,
    )
    dispatch_jokers(ctx_after)

    state.hand.cards = [c for c in state.hand.cards if not c.selected]
    for card in played_cards:
        if not getattr(card, "destroyed", False):
            state.played_cards_this_round.append(card)
            state.played_cards_this_ante.append(card)
            state.discard_pile.append(card)
    state.update_face_down_candidates()
    state.reconcile_face_down_adjacencies()

    if (action.metadata and action.metadata.get("won_round")) or (
        state.target_chips > 0 and state.current_chips >= state.target_chips
    ):
        trigger_end_of_round_effects(state, action)


def _apply_consumable_state(state, card) -> None:
    """Updates consumable-related GameState memory (tarots_used, last_tarot_planet, unique_planets_used, ectoplasm_minus, deck mutations)."""
    if card is None:
        return
    c_type = getattr(card, "type", None)
    c_name = getattr(card, "name", None)
    if c_type == CardType.TAROT:
        state.tarots_used += 1
        if isinstance(c_name, TarotName):
            state.last_tarot_planet = c_name
        if c_name == TarotName.HANGED_MAN:
            selected = [hc.card for hc in state.hand.cards if hc.selected]
            to_destroy = (
                selected[:2] if selected else [hc.card for hc in state.hand.cards[:2]]
            )
            for dc in to_destroy:
                state.remove_card_from_deck(dc, shattered=True)
            state.hand.cards = [
                hc for hc in state.hand.cards if hc.card not in to_destroy
            ]
            state.update_face_down_candidates()
        elif c_name == TarotName.DEATH:
            selected = [hc for hc in state.hand.cards if hc.selected]
            if len(selected) >= 2:
                left, right = selected[0], selected[1]
                source_card = right._real_card if right._real_card else right.card
                if source_card.rank != Rank.NONE:
                    left.resolve_face_down(source_card)
                if right.facing_back:
                    right.reveal_temporarily()
                state.update_face_down_candidates()
        elif c_name == TarotName.STRENGTH:
            selected = [hc for hc in state.hand.cards if hc.selected][:2]
            rank_order = [
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
                Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
            ]
            next_rank = {rank_order[i]: rank_order[(i + 1) % len(rank_order)] for i in range(len(rank_order))}
            for hc in selected:
                if hc.facing_back:
                    if hc._real_card and hc._real_card.rank in next_rank:
                        hc._real_card.rank = next_rank[hc._real_card.rank]
                    elif hc.card.rank in next_rank:
                        hc.card.rank = next_rank[hc.card.rank]
                    hc.reveal_temporarily()
                elif hc.card.rank in next_rank:
                    hc.card.rank = next_rank[hc.card.rank]
            state.update_face_down_candidates()
        elif c_name in (
            TarotName.MAGICIAN,
            TarotName.EMPRESS,
            TarotName.HIEROPHANT,
            TarotName.CHARIOT,
            TarotName.JUSTICE,
            TarotName.TOWER,
            TarotName.LOVERS,
            TarotName.DEVIL,
        ):
            enh_map = {
                TarotName.MAGICIAN: Enhancement.LUCKY,
                TarotName.EMPRESS: Enhancement.MULT,
                TarotName.HIEROPHANT: Enhancement.BONUS,
                TarotName.CHARIOT: Enhancement.STEEL,
                TarotName.JUSTICE: Enhancement.GLASS,
                TarotName.TOWER: Enhancement.STONE,
                TarotName.LOVERS: Enhancement.WILD,
                TarotName.DEVIL: Enhancement.GOLD,
            }
            target_enh = enh_map[c_name]
            max_cards = (
                1
                if c_name
                in (
                    TarotName.CHARIOT,
                    TarotName.JUSTICE,
                    TarotName.TOWER,
                    TarotName.LOVERS,
                    TarotName.DEVIL,
                )
                else 2
            )
            selected = [hc for hc in state.hand.cards if hc.selected][:max_cards]
            for hc in selected:
                hc.card.enhancement = target_enh
                if hc._real_card:
                    hc._real_card.enhancement = target_enh
                if hc.facing_back:
                    hc.reveal_temporarily()
                elif hc.bounds:
                    hc.bounds.possible_enhancements = {target_enh}
            state.update_face_down_candidates()
        elif c_name in (TarotName.SUN, TarotName.MOON, TarotName.STAR, TarotName.WORLD):
            suit_map = {
                TarotName.SUN: Suit.HEARTS,
                TarotName.MOON: Suit.CLUBS,
                TarotName.STAR: Suit.DIAMONDS,
                TarotName.WORLD: Suit.SPADES,
            }
            target_suit = suit_map[c_name]
            selected = [hc for hc in state.hand.cards if hc.selected][:3]
            for hc in selected:
                hc.card.suit = target_suit
                if hc._real_card:
                    hc._real_card.suit = target_suit
                if hc.facing_back:
                    hc.reveal_temporarily()
                elif hc.bounds:
                    hc.bounds.possible_suits = {target_suit}
            state.update_face_down_candidates()
        elif c_name == TarotName.WHEEL_OF_FORTUNE:
            if state.rng:
                oops = get_oops_multiplier(state)
                if state.rng.poll_odds("wheel_of_fortune", 1, 4, oops):
                    eligible = [
                        j for j in state.inventory.jokers if j.edition == Edition.NONE
                    ]
                    if eligible:
                        eligible_sorted = sorted(
                            eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                        )
                        chosen, _ = state.rng.choice(
                            "wheel_of_fortune", eligible_sorted
                        )
                        ed = poll_edition(
                            state.rng, "wheel_of_fortune", no_neg=True, guaranteed=True
                        )
                        chosen.edition = ed
            else:
                state.pending_cv_checks.add("jokers")
        elif c_name in (TarotName.HERMIT, TarotName.TEMPERANCE):
            state.pending_cv_checks.add("money")
    elif c_type == CardType.PLANET:
        if isinstance(c_name, PlanetName):
            state.last_tarot_planet = c_name
        state.unique_planets_used.add(str(c_name))
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.ECTOPLASM:
        state.hand.max_size = max(0, state.hand.max_size - state.ectoplasm_minus)
        state.ectoplasm_minus += 1
        if state.inventory:
            eligible = [
                j for j in state.inventory.jokers if j.edition == Edition.NONE
            ]
            if eligible:
                if state.rng:
                    eligible_sorted = sorted(
                        eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                    )
                    chosen, _ = state.rng.choice("ectoplasm", eligible_sorted)
                    chosen.edition = Edition.NEGATIVE
                else:
                    state.pending_cv_checks.add("jokers")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.HEX:
        if state.inventory:
            eligible = [
                j for j in state.inventory.jokers if j.edition == Edition.NONE
            ]
            if eligible:
                if state.rng:
                    eligible_sorted = sorted(
                        eligible, key=lambda x: getattr(x, "sort_id", 0) or 0
                    )
                    chosen, _ = state.rng.choice("hex", eligible_sorted)
                    chosen.edition = Edition.POLYCHROME
                    for j in list(state.inventory.jokers):
                        if j != chosen and not getattr(j, "eternal", False):
                            state.inventory.jokers.remove(j)
                else:
                    state.pending_cv_checks.add("jokers")
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.CRYPTID:
        selected = [c for c in state.hand.cards if c.selected]
        target_hc = (
            selected[0]
            if selected
            else (state.hand.cards[0] if state.hand.cards else None)
        )
        if target_hc:
            if target_hc.facing_back:
                target_hc.reveal_temporarily()
            source = target_hc._real_card or target_hc.card
            state.add_card_to_deck(copy.deepcopy(source), add_to_hand=True)
            state.add_card_to_deck(copy.deepcopy(source), add_to_hand=True)
        else:
            state.deck_cards_count += 2
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.AURA:
        selected = [hc for hc in state.hand.cards if hc.selected][:1]
        if selected:
            target_hc = selected[0]
            ed = Edition.POLYCHROME
            if state.rng:
                ed = poll_edition(state.rng, "aura", no_neg=True, guaranteed=True)
            target_hc.card.edition = ed
            if target_hc._real_card:
                target_hc._real_card.edition = ed
            if target_hc.facing_back:
                target_hc.reveal_temporarily()
            state.update_face_down_candidates()
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.SIGIL:
        suit = Suit.SPADES
        if state.rng:
            suit = state.rng.choice("sigil", [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS])[0]
        for hc in state.hand.cards:
            hc.card.suit = suit
            if hc._real_card:
                hc._real_card.suit = suit
            if hc.facing_back:
                hc.reveal_temporarily()
            elif hc.bounds:
                hc.bounds.possible_suits = {suit}
        state.update_face_down_candidates()
    elif c_type == CardType.SPECTRAL and c_name in (
        SpectralName.GRIM,
        SpectralName.FAMILIAR,
        SpectralName.INCANTATION,
    ):
        if state.rng:
            hand_cards = [hc.card for hc in state.hand.cards]
            victim = predict_random_destroy(state.rng, hand_cards)
            if victim is not None:
                state.remove_card_from_deck(victim)
            if c_name == SpectralName.FAMILIAR:
                new_cards = predict_familiar_spawns(state.rng)
            elif c_name == SpectralName.GRIM:
                new_cards = predict_grim_spawns(state.rng)
            else:
                new_cards = predict_incantation_spawns(state.rng)
            for nc in new_cards:
                state.add_card_to_deck(nc, add_to_hand=True)
        else:
            victim = state.hand.cards[0].card if state.hand.cards else None
            if victim is not None:
                state.remove_card_from_deck(victim)
                state.hand.cards.pop(0)
            else:
                state.remove_card_from_deck(
                    PlayingCard(rank=Rank.ACE, suit=Suit.SPADES)
                )
            spawn_count = (
                2
                if c_name == SpectralName.GRIM
                else (3 if c_name == SpectralName.FAMILIAR else 4)
            )
            spawn_rank = (
                Rank.ACE
                if c_name == SpectralName.GRIM
                else (Rank.KING if c_name == SpectralName.FAMILIAR else Rank.TWO)
            )
            for _ in range(spawn_count):
                state.add_card_to_deck(
                    PlayingCard(
                        rank=spawn_rank, suit=Suit.SPADES, enhancement=Enhancement.BONUS
                    ),
                    add_to_hand=True,
                )
    elif c_type == CardType.SPECTRAL and c_name == SpectralName.IMMOLATE:
        if state.rng:
            hand_cards = [hc.card for hc in state.hand.cards]
            destroyed = predict_immolate_destructions(state.rng, hand_cards, count=5)
            for dc in destroyed:
                state.remove_card_from_deck(dc)
            state.money += 20
        else:
            to_destroy = [hc.card for hc in state.hand.cards[:5]]
            for dc in to_destroy:
                state.remove_card_from_deck(dc)
            for _ in range(max(0, 5 - len(to_destroy))):
                state.remove_card_from_deck(
                    PlayingCard(rank=Rank.ACE, suit=Suit.SPADES)
                )
            state.hand.cards = state.hand.cards[5:]
            state.money += 20
    elif c_type == CardType.SPECTRAL and c_name in (
        SpectralName.TALISMAN,
        SpectralName.DEJA_VU,
        SpectralName.TRANCE,
        SpectralName.MEDIUM,
    ):
        seal_map = {
            SpectralName.TALISMAN: Seal.GOLD,
            SpectralName.DEJA_VU: Seal.RED,
            SpectralName.TRANCE: Seal.BLUE,
            SpectralName.MEDIUM: Seal.PURPLE,
        }
        target_seal = seal_map[c_name]
        selected = [hc for hc in state.hand.cards if hc.selected][:1]
        for hc in selected:
            hc.card.seal = target_seal
            if hc._real_card:
                hc._real_card.seal = target_seal
            if hc.facing_back:
                hc.reveal_temporarily()
            elif hc.bounds:
                hc.bounds.possible_seals = {target_seal}
        state.update_face_down_candidates()


def on_action(state, action: Action):
    """Executes deterministic game engine state transitions and triggers."""
    # Implicit joker reordering: if the action specifies a joker order, apply it before evaluating the action!
    j_order = action.joker_order or (action.metadata and action.metadata.get("joker_order"))
    if j_order and state.inventory and state.inventory.jokers:
        current_jokers = state.inventory.jokers
        if len(j_order) == len(current_jokers) and all(
            isinstance(x, int) and 0 <= x < len(current_jokers) for x in j_order
        ):
            state.inventory.jokers = [current_jokers[i] for i in j_order]
        else:
            by_id = {j.id: j for j in current_jokers if j.id != 0}
            if all(jid in by_id for jid in j_order):
                state.inventory.jokers = [by_id[jid] for jid in j_order]
        sync_dynamic_joker_states(state)

    match action.type:
        case ActionType.SELECT_CARD:
            if action.target_idx is not None:
                state.hand.select(action.target_idx)

        case ActionType.SORT_RANK:
            state.apply_sort_observation(
                sort_type="rank",
                selected_slot_uid=action.metadata.get("selected_slot_uid")
                if action.metadata
                else None,
            )

        case ActionType.SORT_SUIT:
            state.apply_sort_observation(
                sort_type="suit",
                selected_slot_uid=action.metadata.get("selected_slot_uid")
                if action.metadata
                else None,
            )

        case ActionType.PLAY_HAND:
            state.hands_left = max(0, state.hands_left - 1)
            state.hands_played_this_round += 1

            for c in state.hand.cards:
                if c.selected and c.facing_back and c._real_card:
                    c.resolve_face_down(c._real_card, lift_facing_back=True)

            played_cards = [c.card for c in state.hand.cards if c.selected]
            if action.metadata and action.metadata.get("defer_play_for_hook"):
                state.hand.cards = [c for c in state.hand.cards if not c.selected]
                state._hook_pending_cards = played_cards
            else:
                _evaluate_played_cards(state, action, played_cards)

        case ActionType.DISCARD:
            is_hook = bool(action.metadata and action.metadata.get("hook"))
            if not is_hook:
                state.discards_left = max(0, state.discards_left - 1)
                state.discards_used_this_round += 1

            for c in state.hand.cards:
                if c.selected and c.facing_back and c._real_card:
                    c.resolve_face_down(c._real_card, lift_facing_back=True)

            discarded_cards = [c.card for c in state.hand.cards if c.selected]
            for card in discarded_cards:
                if (
                    card.seal == Seal.PURPLE
                    and not getattr(card, "debuffed", False)
                    and state.inventory
                    and state.inventory.can_add_consumable
                ):
                    if state.rng:
                        pred = predict_purple_seal_tarot(state.rng, ante=state.ante)
                        state.inventory.consumables.append(pred.to_tarot_card())
                    else:
                        state.pending_cv_checks.add("consumables")

            ctx = TriggerContext(
                event=TriggerEvent.DISCARD,
                state=state,
                action=action,
                full_hand=discarded_cards,
            )
            dispatch_jokers(ctx)

            state.hand.cards = [c for c in state.hand.cards if not c.selected]
            for card in discarded_cards:
                if not getattr(card, "destroyed", False):
                    state.discard_pile.append(card)

            state.update_face_down_candidates()
            state.reconcile_face_down_adjacencies()

            pending_hook_cards = getattr(state, "_hook_pending_cards", None)
            if is_hook and pending_hook_cards is not None:
                state._hook_pending_cards = None
                _evaluate_played_cards(state, action, pending_hook_cards)

        case ActionType.REROLL_SHOP:
            if state.free_rerolls > 0:
                state.free_rerolls -= 1
            else:
                state.money -= state.reroll_cost
                state.reroll_cost += 1

            ctx = TriggerContext(
                event=TriggerEvent.REROLL_SHOP, state=state, action=action
            )
            dispatch_jokers(ctx)

        case ActionType.BUY_CARD:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_cards
            ):
                card = state.shop_cards.pop(action.target_idx)
                cost = card.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                )
                state.money -= cost
                extra = {}
                if card.type == CardType.JOKER:
                    state.inventory.jokers.append(card)
                    on_joker_added(state, card)
                elif card.type in (CardType.TAROT, CardType.PLANET, CardType.SPECTRAL):
                    state.inventory.consumables.append(card)
                elif card.type == CardType.PLAYING_CARD:
                    state.add_card_to_deck(card, add_to_hand=False)
                    extra["already_added_to_deck"] = True
                ctx = TriggerContext(
                    event=TriggerEvent.CARD_BOUGHT,
                    state=state,
                    action=action,
                    target_card=card,
                    extra=extra,
                )
                dispatch_jokers(ctx)

        case ActionType.BUY_AND_USE:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_cards
            ):
                card = state.shop_cards.pop(action.target_idx)
                cost = card.get_cost(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                )
                state.money -= cost
                _apply_consumable_state(state, card)
                ctx = TriggerContext(
                    event=TriggerEvent.USE_CONSUMABLE,
                    state=state,
                    action=action,
                    target_card=card,
                )
                dispatch_jokers(ctx)

        case ActionType.USE_CONSUMABLE:
            used_card = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.inventory.consumables
            ):
                used_card = state.inventory.consumables.pop(action.target_idx)
            if used_card is not None:
                _apply_consumable_state(state, used_card)
                ctx = TriggerContext(
                    event=TriggerEvent.USE_CONSUMABLE,
                    state=state,
                    action=action,
                    target_card=used_card,
                )
                dispatch_jokers(ctx)

        case ActionType.OPEN_PACK:
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_packs
            ):
                pack = state.shop_packs.pop(action.target_idx)
                state.money -= pack.cost
                state.active_pack = pack
                state.pack_choices_remaining = pack.choose
                state.location = Location.PACK
                ctx = TriggerContext(
                    event=TriggerEvent.OPEN_BOOSTER, state=state, action=action
                )
                dispatch_jokers(ctx)

        case ActionType.USE_IN_PACK:
            chosen_card = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.pack_cards
            ):
                chosen_card = state.pack_cards.pop(action.target_idx)
            elif action.metadata and isinstance(action.metadata, dict):
                meta_set = action.metadata.get("set")
                meta_card = action.metadata.get("card")
                if meta_set in ("Default", "Enhanced"):
                    state.deck_cards_count += 1
                elif meta_card == "Cryptid":
                    state.deck_cards_count += 2

            if chosen_card is not None:
                if chosen_card.type == CardType.JOKER:
                    state.inventory.jokers.append(chosen_card)
                    on_joker_added(state, chosen_card)
                elif chosen_card.type == CardType.PLAYING_CARD:
                    state.add_card_to_deck(chosen_card, add_to_hand=False)
                    ctx = TriggerContext(
                        event=TriggerEvent.CARD_BOUGHT,
                        state=state,
                        action=action,
                        target_card=chosen_card,
                        extra={"already_added_to_deck": True},
                    )
                    dispatch_jokers(ctx)
                else:
                    _apply_consumable_state(state, chosen_card)
                    ctx = TriggerContext(
                        event=TriggerEvent.USE_CONSUMABLE,
                        state=state,
                        action=action,
                        target_card=chosen_card,
                    )
                    dispatch_jokers(ctx)
                state.pack_choices_remaining = max(0, state.pack_choices_remaining - 1)
                if state.pack_choices_remaining == 0 or not state.pack_cards:
                    state.active_pack = None
                    state.pack_cards.clear()
                    state.location = Location.SHOP

        case ActionType.SKIP_PACK:
            ctx = TriggerContext(
                event=TriggerEvent.SKIP_BOOSTER, state=state, action=action
            )
            dispatch_jokers(ctx)
            state.active_pack = None
            state.pack_cards.clear()
            state.pack_choices_remaining = 0
            state.location = Location.SHOP

        case ActionType.REDEEM_VOUCHER:
            voucher = None
            if action.target_idx is not None and 0 <= action.target_idx < len(
                state.shop_vouchers
            ):
                voucher = state.shop_vouchers.pop(action.target_idx)
            elif action.metadata and isinstance(action.metadata, dict):
                v_str = action.metadata.get("card")
                for vn in VoucherName:
                    if vn.value == v_str:
                        voucher = Voucher(name=vn)
                        break
            if voucher is not None:
                state.money -= voucher.price
                if voucher.name not in state.vouchers_redeemed:
                    state.vouchers_redeemed.append(voucher.name)
                match voucher.name:
                    case VoucherName.GRABBER | VoucherName.NACHO_TONG:
                        state.hands_left += 1
                    case VoucherName.WASTEFUL | VoucherName.RECYCLOMANCY:
                        state.discards_left += 1
                    case VoucherName.PAINT_BRUSH | VoucherName.PALETTE:
                        state.hand.max_size += 1
                    case VoucherName.ANTIMATTER:
                        state.inventory.joker_slots += 1
                    case VoucherName.CRYSTAL_BALL:
                        state.inventory.consumable_slots += 1
                    case VoucherName.REROLL_SURPLUS | VoucherName.REROLL_GLUT:
                        state.reroll_cost = max(0, state.reroll_cost - 2)
                    case VoucherName.HIEROGLYPH:
                        state.ante = max(1, state.ante - 1)
                        state.hands_left = max(1, state.hands_left - 1)
                    case VoucherName.PETROGLYPH:
                        state.ante = max(1, state.ante - 1)
                        state.discards_left = max(0, state.discards_left - 1)

        case ActionType.SELL_CARD:
            sold_card = None
            if action.target_idx is not None:
                if action.target_idx < len(state.inventory.jokers):
                    sold_card = state.inventory.jokers.pop(action.target_idx)
                else:
                    c_idx = action.target_idx - len(state.inventory.jokers)
                    if 0 <= c_idx < len(state.inventory.consumables):
                        sold_card = state.inventory.consumables.pop(c_idx)
            if sold_card:
                state.money += sold_card.get_sell_value(
                    discount_percent=state.discount_percent,
                    inflation=state.inflation,
                )
                if sold_card.type == CardType.JOKER:
                    on_joker_removed(state, sold_card)
                    if state.current_blind == BlindName.VERDANT_LEAF:
                        state.boss_disabled = True
                        for j in state.inventory.jokers:
                            if not (j.perishable and j.perish_tally <= 0):
                                j.debuffed = False
                ctx = TriggerContext(
                    event=TriggerEvent.CARD_SOLD,
                    state=state,
                    action=action,
                    target_card=sold_card,
                )
                dispatch_jokers(ctx)

        case ActionType.CASH_OUT:
            if not getattr(state, "round_won", False):
                trigger_end_of_round_effects(state, action)

            if action.metadata and action.metadata.get("round_dollars") is not None:
                state.money += action.metadata["round_dollars"]
            else:
                if action.metadata and action.metadata.get("include_round_payout"):
                    state.money += calculate_base_round_payout(state)
                ctx = TriggerContext(
                    event=TriggerEvent.END_OF_ROUND,
                    state=state,
                    action=action,
                    extra={"phase": "cash_out_dollars"},
                )
                dispatch_jokers(ctx)
            state.round_won = False
            state.location = Location.SHOP
            base_reroll = 5
            if VoucherName.REROLL_SURPLUS in state.vouchers_redeemed:
                base_reroll = max(0, base_reroll - 2)
            if VoucherName.REROLL_GLUT in state.vouchers_redeemed:
                base_reroll = max(0, base_reroll - 2)
            if SkipTag.D6_TAG in state.active_tags:
                state.active_tags.remove(SkipTag.D6_TAG)
                base_reroll = 0
            state.reroll_cost = base_reroll

        case ActionType.NEXT_ROUND:
            ctx = TriggerContext(
                event=TriggerEvent.LEAVE_SHOP, state=state, action=action
            )
            dispatch_jokers(ctx)
            state.location = Location.SELECT

        case ActionType.SELECT_BLIND:
            state.round_won = False
            state.location = Location.BLIND
            if state.boss_blind is None and (state.seed or getattr(state, "rng", None)):
                if hasattr(state, "roll_boss_blind"):
                    state.roll_boss_blind(is_reroll=False)
            state.hands_played_this_round = 0
            state.discards_used_this_round = 0

            base_hands = 4
            if VoucherName.GRABBER in state.vouchers_redeemed:
                base_hands += 1
            if VoucherName.NACHO_TONG in state.vouchers_redeemed:
                base_hands += 1
            if VoucherName.HIEROGLYPH in state.vouchers_redeemed:
                base_hands -= 1
            for j in state.inventory.jokers:
                if j.name == JokerName.TROUBADOUR and not j.debuffed:
                    base_hands -= 1
            state.hands_left = max(1, base_hands)

            base_discards = 3
            if VoucherName.WASTEFUL in state.vouchers_redeemed:
                base_discards += 1
            if VoucherName.RECYCLOMANCY in state.vouchers_redeemed:
                base_discards += 1
            if VoucherName.PETROGLYPH in state.vouchers_redeemed:
                base_discards -= 1
            for j in state.inventory.jokers:
                if not j.debuffed:
                    if j.name == JokerName.DRUNKARD:
                        base_discards += 1
                    elif j.name == JokerName.MERRY_ANDY:
                        base_discards += 3
            state.discards_left = max(0, base_discards)

            state.discard_pile.clear()
            state.played_cards_this_round.clear()
            state.hands_played_this_round_by_type.clear()
            state.locked_hand_type = None
            state.boss_disabled = False

            ctx = TriggerContext(
                event=TriggerEvent.SETTING_BLIND,
                state=state,
                action=action,
                extra={"initial_jokers": list(state.inventory.jokers)},
            )
            dispatch_jokers(ctx)

        case ActionType.SKIP_BLIND:
            state.blinds_skipped += 1
            if state.current_blind == BlindName.SMALL:
                if state.small_blind_tag is not None:
                    state.active_tags.append(state.small_blind_tag)
                state.current_blind = BlindName.BIG
            elif state.current_blind == BlindName.BIG:
                if state.big_blind_tag is not None:
                    state.active_tags.append(state.big_blind_tag)
                if state.boss_blind is not None:
                    state.current_blind = state.boss_blind

        case ActionType.REROLL_BOSS:
            state.money -= 10
            if hasattr(state, "roll_boss_blind"):
                state.roll_boss_blind(is_reroll=True)
            elif state.seed or getattr(state, "rng", None):
                if getattr(state, "rng", None) is None and state.seed:
                    state.set_seed(state.seed)
                try:
                    from game.engine.rng_sim.events import get_next_boss_blind
                except ImportError:
                    from engine.rng_sim.events import get_next_boss_blind
                if not hasattr(state, "bosses_used") or state.bosses_used is None:
                    state.bosses_used = {}
                pred = get_next_boss_blind(
                    state.rng,
                    ante=state.ante,
                    bosses_used=state.bosses_used,
                    is_reroll=True,
                )
                state.boss_blind = pred.blind_name

        case ActionType.REORDER_JOKERS:
            order = action.joker_order or (action.metadata and action.metadata.get("joker_order"))
            if order:
                current_jokers = state.inventory.jokers
                if len(order) == len(current_jokers) and all(isinstance(x, int) and 0 <= x < len(current_jokers) for x in order):
                    state.inventory.jokers = [current_jokers[i] for i in order]
                else:
                    by_id = {j.id: j for j in current_jokers if j.id != 0}
                    if all(jid in by_id for jid in order):
                        state.inventory.jokers = [by_id[jid] for jid in order]

        case ActionType.REORDER_HAND:
            order = action.metadata and action.metadata.get("hand_order")
            if order:
                current_hand = state.hand.cards
                if len(order) == len(current_hand) and all(isinstance(x, int) and 0 <= x < len(current_hand) for x in order):
                    state.hand.cards = [current_hand[i] for i in order]
                else:
                    by_id = {c.card.id: c for c in current_hand if c.card and c.card.id != 0}
                    if all(cid in by_id for cid in order):
                        state.hand.cards = [by_id[cid] for cid in order]
                state.reconcile_face_down_adjacencies()

    sync_dynamic_joker_states(state)
