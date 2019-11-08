# strassen3by3
A repo that has a GA that can be used for the solution to strassen 3by3 multiplication.


<p> There were several memory issues with my original expand method, so I replaced it with the
basic element multiplication.<p>



## Parameters

<p> you can play around with the parameters that are on line 499- 505 in the strassen_search_light.py file. I would suggest playing around with 2 dimension, 7 multiplications first to get an idea of the effects.  I print the best cost at the end of every generation to give you an idea of how the 
cost is progressing. <p>

>    multiplications = 7
>    matrix_dimension = 2
>    pop = 40  # np.random.randint(10, 21)
>    mutation_rate = 14  # np.random.randint(35, 45)
>    runs = 250
>    purge = .40
>    purge_best = True


<p>  If you run with dimension of 3 and 23 multipications, you will see that 240 dimensions likely isn't enough <p>