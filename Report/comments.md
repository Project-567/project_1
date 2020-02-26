# Comments

- 1c: your P is presented for a fixed (random) policy: in MDP we have one transition matrix per action, so for full marks you need to show where the non-zero entries of these matrices are [-1] 

- 1d: what would happen if you change the penalty associated with going off the grid (e.g. to -2)? [-1] 

- 1e: your Bellman equation appears to be evaluating a fixed policy: it should instead contain a maximization step over actions [-1] 

- in the final part of your write-up part c: you didn't discuss the relative time/space complexity associated with policy evaluation/improvement steps. in the best case policy evaluation requires |S|^3 steps per iteration and inverting a |S|-by-|S| matrix, whereas VI requires |S|^2 |A| per iteration: on a per-iteration basis, VI is more "efficient" than PI for larger |S|, even though you are correct to point out that PI can in this case (but not necessarily always) converge faster. [-1]