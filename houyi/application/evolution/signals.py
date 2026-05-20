from __future__ import annotations

from houyi.application.evolution.events import EvolutionEvent, EvolutionEventType, EvolutionSignal


class EvolutionSignalMiner:
    def mine(self, events: list[EvolutionEvent]) -> list[EvolutionSignal]:
        signals: list[EvolutionSignal] = []
        for event in events:
            if event.event_type == EvolutionEventType.RECALL_FAILURE:
                signals.append(
                    EvolutionSignal(
                        signal_type="recall_failure",
                        target=event.target,
                        severity=event.metrics.get("severity", 1.0),
                        event_ids=(event.event_id,),
                    )
                )
            elif event.event_type == EvolutionEventType.USER_CORRECTION:
                signals.append(
                    EvolutionSignal(
                        signal_type="user_correction",
                        target=event.target,
                        severity=event.metrics.get("severity", 0.8),
                        event_ids=(event.event_id,),
                    )
                )
        return signals
