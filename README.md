<H1>Markov Chain Characteristics in Discrete Time</H1>
<b></b>This repository implements the characteristics of Markov chains in discrete time. It covers the following aspects:</b>

Characteristics Covered
* Irreducibility: The Markov chain is irreducible if every state is accessible from every other state.
* Reducibility: The Markov chain is reducible if it can be partitioned into two or more disjoint sets such that there are no transitions between the sets.
* Recurrent States: A state is recurrent if, once the chain enters that state, it will return to that state with probability 1.
* Transient States: A state is transient if, once the chain enters that state, it may never return.
* Periodic States: A state is periodic if the chain can return to the state only at multiples of some integer greater than 1.
* Aperiodic States: A state is aperiodic if the chain can return to the state at any time.
* Absorbing States: A state is absorbing if it is impossible to leave that state once entered.
* CMTD verification :This repository also provides tools for verifying the Chapman–Kolmogorov forward equation (CMTD) for the Markov chain.
