(define (problem test-1)
  (:domain test-domain-gridworld)
  (:objects
    s1 - location
    s2 - location
    s3 - location
  )
  (:init
    (robot-at s1)

    (conn-prob s1 s2 right)
    (conn-prob s2 s1 left)

    (conn-prob s2 s3 right)
    (conn-prob s3 s2 left)
  )
  (:goal (and (robot-at s3)))
)
