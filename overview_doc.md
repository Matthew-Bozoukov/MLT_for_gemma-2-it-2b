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
