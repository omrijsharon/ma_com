# ma_com
Multi-Agent COMmunication module for pytorch.

## What does it do?
This module takes in messages with length of 'msg_dim' from a number of agents ('n_agents') and outputs a single message that "summerizes" all the input messages. The output will always be the size of 'msg_dim', independent of the number of agents.

## What is so special about it?
1. In the case of a single agent (self-interaction / self-communication), the encoded input and the encoded output are identical - as expected, since there is no interaction with other agents. Furthermore, the encoder-decoder networks that the user inputs can be trained this way to be the inverse of each other.
2. The output is independent of the number of agents in the input and the order of the agents' messages doesn't matter. This module is order-invariant.
This takes [DeepSets](https://github.com/manzilzaheer/DeepSets) to the next level.

## How does it do this?
Borrowing from the attention mechanism: 
1. Every message is passed through a single "encoder" network (in contrast from the query, key, value networks in attention).
2. All the encoded messages are stacked up to form a matrix with size of [n_agents, encoded_msg_dim].
3. We take the outer product of this [n_agents, encoded_msg_dim] matrix with itself:
* [n_agents, encoded_msg_dim] @ [encoded_msg_dim, n_agents] = [n_agents, n_agents].
4. Softmaxing the columns of this matrix.
5. Dot product of this new [n_agents, n_agents] with the original encoded messages matrix:
* [n_agents, n_agents] @ [n_agents, encoded_msg_dim] = [n_agents, encoded_msg_dim]
6. Decoding this new encoded messages matrix to get [n_agents, msg_dim]
7. Reducing the n_agents dimension by mean or max operation over the n_agents dimension to get a single vector [msg_dim].

All this operations are done for a batch. So batch_size should be added to the first dimention of every matrix. Yet, it is easier to explain and to understand it without the batch_size.

## How to use it? Example:
```
    batch_size = 20
    n_agents = 10
    msg_dim = 4
    latent_msg_dim = 2
    x = torch.randn(batch_size, n_agents, msg_dim)
    macom = Macom(
        encoding_net=nn.Linear(msg_dim, latent_msg_dim),
        decoding_net=nn.Linear(latent_msg_dim, msg_dim)
    )
    output = macom(x)
```
In this case, output size will be [20, 4], or in the for a general case [batch_size, msg_dim]. Independent of the number of agents.
