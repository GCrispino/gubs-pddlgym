(define (domain test-domain-gridworld)
    (:requirements :typing :probabilistic-effects)
    (:types location direction)

    (:constants
        left - direction
        right - direction
        down - direction
        up - direction)

    (:predicates
        (robot-at ?v0 - location)
        (conn ?v0 - location ?v1 - location ?v2 - direction)
        (conn-prob ?v0 - location ?v1 - location ?v2 - direction)
        (move ?v0 - direction)
    )

    ; (:actions move)

    (:action move
        :parameters (?from - location ?to - location ?dir - direction)
        :precondition (and (robot-at ?from) (conn ?from ?to ?dir) (move ?dir))
        :effect (and
            (not (robot-at ?from))
            (robot-at ?to))
    )

    (:action move-prob
        :parameters (?from - location ?to - location ?dir - direction)
        :precondition (and (robot-at ?from) (conn-prob ?from ?to ?dir) (move ?dir))
        :effect (and
            (not (robot-at ?from)) 
            (probabilistic
              0.5 (robot-at ?from)
              0.5 (and (robot-at ?to))
            ))
    )
)

