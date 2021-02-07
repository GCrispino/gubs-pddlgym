(define (problem test-1)
  (:domain test-domain-gridworld)
  (:objects
    s1 - location
    s2 - location
    s3 - location
    s4 - location
    s5 - location
    s6 - location
    s7 - location
    s8 - location
    s9 - location
    s10 - location
  )
  (:init
    (robot-at s1)

    (conn-prob s1 s2 right)
    (conn-prob s1 s6 down)

    (conn-prob s2 s3 right)
    (conn-prob s2 s7 down)
    (conn-prob s2 s1 left)

    (conn-prob s3 s4 right)
    (conn-prob s3 s8 down)
    (conn-prob s3 s2 left)

    (conn-prob s4 s5 right)
    (conn-prob s4 s9 down)
    (conn-prob s4 s3 left)

    (conn-prob s5 s10 down)
    (conn-prob s5 s4 left)

    (conn s6 s7 right)
    (conn s6 s1 up)

    (conn s7 s8 right)
    (conn s7 s6 left)
    (conn s7 s2 up)

    (conn s8 s9 right)
    (conn s8 s7 left)
    (conn s8 s3 up)

    (conn s9 s10 right)
    (conn s9 s8 left)
    (conn s9 s4 up)

    (conn s10 s9 left)
    (conn s10 s5 up)
  )
  (:goal (and (robot-at s5)))
)
