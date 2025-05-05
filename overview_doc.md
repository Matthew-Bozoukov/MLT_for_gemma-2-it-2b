A small overview of attribution graphs and their construction

# Model Setup:
The graph requires a local replacement model, just using the CLTs outputs rather than the MLPs outputs, but still needs the MLP outputs to be computed. Other than that the rest of the LLM can be formatted however is needed, and the local replacement model is generally not necessary as some parts of the graph just need to be removed in a simple change.

# Graph Structure:
The Attribution Graph has 4 types of nodes. Output nodes, those at the end of the graph, correspond to candidate output tokens. The intermediate nodes correspond to CLT activations at each layer and context position. The primary input nodes, those at the start of the graph, correspond to the embeddings of the prompt tokens. The error nodes correspond to the error between the CLT and original MLP activations at each intermediate node.

The edges of the graph defined $A_{s\rightarrow t}$ between feature node $s$ and target node $t$ uses a theoretical fully connected network, with $a_s$ being the activation of the original feature $s$ and $w_{s\rightarrow t}$ being that theoretical network connection.

$$
A_{s\rightarrow t}=a_sw_{s\rightarrow t}
$$

# Graph Creation:
Since the structure views the connection between each node as a simple scalar multiplication, and because LLMs are much more complex than scalar multiplications, the computation for the values gets a little complex. The equation for them is below (the first is from feature node to feature node, and the second is the emedding to feature weights), but it basically boils down to the some element of the previous node (output) times a Jacobian times some element of the current node (input). The jacobian acts to denote the change in data or the data progression through the model.

$$
\begin{gather}
A_{s\rightarrow t}=a_sw_{s\rightarrow t}=a_s\sum_{l_s\leq l\leq l_t}(W^{l_s\rightarrow l}_{\text{dec},s})J^\bigtriangledown_{c_s,l\rightarrow c_t,l_t}W^{l_t}_{\text{enc},t}\\
w_{s\rightarrow t,\text{edge}}=\text{Emb}^T_sJ^\bigtriangledown_{c_s,l_s\rightarrow c_t,l_t}W^{l_t}_{\text{enc},t}
\end{gather}
$$

These computations are then formalized for each type of neuron. The output nodes have an input of $v_\text{in}=\Delta(\text{logit}_\text{tok}-\overline{\text{logit}})$ that is the gradient of the difference between the pre-softmax logits and the mean of all the logits. A feature node that corresponds to a feature $s$ and context position $c_s$ has an output $v^l_\text{out}=W^{l_s\rightarrow l}_{\text{dec},s}$ and an input $v^l_\text{in}=W^{l_s}_{\text{enc},s}$, which are the CLT decoder output and the CLT encoder output for that feature respectively, or at least I'm assuming they're very vague on what they actually mean by those. Embedding nodes have outputs $v_\text{out}=\text{Emb}_\text{tok}$ that correspond to the embedding of their respective input tokens. The error nodes outputs $v_\text{out}=\text{MLP}_l(x_{c,l})-\text{CLT}_l(x_c)$ correspond to the difference between the MLP output and the CLT output for any specific context position $c$. These then allow the formulation of the graph connections to change, with the first below being from embedding or error nodes to feature or outputs nodes and the second being from feature nodes to feature or output nodes. The first boils down to an output times the Jacobian times the input and the second boils down to the same thing repeated over every layer in between both nodes multiplied by the activation of the previous node's element.

$$
\begin{gather}
A_{s\rightarrow t}=v^T_{\text{out},s}J^\bigtriangledown_{c_s,l_s\rightarrow c_t,l_t}v_{\text{in},t}\\
A_{s\rightarrow t}=a_s\sum_{l_s\leq l\leq l_t}(v^l_\text{out})^TJ^\bigtriangledown_{c_s,l\rightarrow c_t,l_t}v_{\text{in},t}
\end{gather}
$$

The jacobian is the sum over all the paths, including both attention heads and residual connection. This is formalied as a sum of transformations $\pi_i$ through the set $\mathcal{P}(c_s,c_t,l_s,l_t)$, the set of all paths starting at context position $c_s$ in layer $l_s$ and ending at position $c_t$ in layer $l_t$. The transformation $\pi_i$ is either the identity (if thats not a common term then I don't know this is what it says verbatim, but it is just how the data progresses through the step) for a residual stream step and $a^{h_i}_{c_i\rightarrow c_{i+1}}OV_{h_i}$ (the attention weight from position $c_i$ to $c_{i+1}$ multiplied by transformation $OV_{h_i}$ for attention head $h_i$) for attention steps.

$$
\begin{gather}
J^\bigtriangledown_{c_s,l_s\rightarrow c_t,l_t}=\sum_{p\in\mathcal{P}(c_s,c_t,l_s,l_t)}\pi_p\\
\pi_p=\prod_i\pi_i
\end{gather}
$$

# Graph Pruning:
To make the graph interpretable the graph needs to be heavily pruned, where embedding nodes and error nodes are the only nodes safe. This is done in a two step process that prunes nodes and then the edges between them. First the adjacency matrix of the graph is generated and the absolute value of each edge weight is taken. The matrix is then normalized along the inputs of each node to sum to 1, resulting in a matrix $A$. The indirect influence matrix $B$, a matrix that indicates the strength of all paths between a pair of nodes, is then derived $B=A+A^2+A^3+\dots=(I-A)^{-1}-I$. The weighted average of the values of $B$ corresponding to logit nodes are then used to generate a set of logit influence scores for each node in the graph. These scores are sorted and a minimum cutoff index (a maximum distance to which the nodes can reach the threshold, with a value of 0.8 working for them, where the value compared to the threshold is he sum of the influence scores divided by the total sum) is chosen, with the non-logit nodes that don't make the threshold by the cutoff being pruned.

The same matrix $B$ is generated, but this time a score is assigned to each edge by multiplying the logit influence score of the edge's output by the normalized edge weight. The edges are pruned with the same cutoff threshold strategy as the nodes, with a threshold of 0.98 working for them.

Lastly, to prune logit nodes, the logit nodes that correspond to the top K most likely token outputs are kept and the rest are pruned, as well as having the total probability of the logit nodes not exceeding 0.95 (if they do, K is clamped to 10).
