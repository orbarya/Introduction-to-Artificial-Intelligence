20111660
30048639
*****
Comments:
Our better evaluation function gives weights to each tile of the board and point wise multiplies the game state with the weights.

We tried two forms of weights: 
1. The right-most lowest corner takes the biggest score and the scores decrease as tiles get farther away from this tile in manhattan distance.
This is summed with another set of weights for the tiles of the lowest row, i.e each tile in the lowest
2. Snake decreasing order from the lowest right-most corner and decreasing row first:
The following might explain better, the weights are decreasing in the direction of the arrows.
>>>>>>>>>
<<<<<<<<<
>>>>>>>>>
<<<<<<<<<
