this repository is for my internal simulation package.
Large parts of this package was made or remade using AI.
But much of the work builds on 3 different pieces of work I've been making on my own previously.
The structure is as I've always intended. The focus is simulations, so the idea that a stochastic processes package,
population dynamics, finance and physics package are subpackages was always my intention.
the architecture, the package setups, the ideas leading to the code and the command setups were all designed by me in hand.
If asked I may blunder for a moment how each sub piece is built and how each exact equation was set up.
But asking in broad, you will find that I understand and know each piece of this.

My brain functions as a "focus and release" mechanism for this, designing details of each process down to the equations, parameters, dynamics
for example governing how one individual in my SIR model differs from another based on "sociability", "infectiousness", "number of friends/family in network", "health recovery speed", "prior resistance" and so on.
To those I must also design and decide how those parameters plug into infection spread, recovery, setup design. I would go over if I should map it as a agent model or as a collective model.
I want to get as close to reality as possible, without tanking the user's computer power. But at the same time I want to test against real world data. I need to consider how that should be tested.
Where to get data, whether I can divide parts of the process and test them separately on process power requirements, accuracy to reality, accuracy to intent. 

I sadly no longer sit around with much of the leg work, but I instead sit around with a lot of the decision making, creativity, testing and architecture.

For both physics and finance parts of the package multiple more pieces will be coming along later, as that both of those will be tied into other projects.
As much as I would love to deep dive into economics, population dynamics and more, to properly research those processes and find even more realistic behaviors due to corrections in dynamics, equations, behavior mappings for indiviuals and better architecture. Those will be put on hold until I have genuine time to do nothing at all.

For finance I will be working on setting up versions of the market model that are closer to reality, I will have to tailor make agent behaviors to map better onto reality, this will require understanding trader behaviors, investor behaviors, financial institution behaviors better, but also mapping out events in the market better. 
One part I am playing around with is moving some work into C++, so as to speed up and utilise better speed than python. The improvements made here will also be tied in with [Tomorrow markets](https://tomorrowmarkets.github.io/)

## physics
This has been updated using AI, but is based on my work with astrophysics and particle swarms I've worked on previously.
the animations, the work on particle setups, the idea about polarity, spin and energy resulting, has all been come from 
