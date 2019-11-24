class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
		super(DecoderRNN, self).__init__()
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size

		#lstm cell
		self.lstm_cell = nn.LSTMCell(input_size = embed_size, hidden_size = hidden_size)
		#output fully connected layer
		self.fc_out = nn.Linear(in_features = hidden_size, out_features = vocab_size)
		#embeddding layer 
		self.embed = nn.Embedding(num_embedding = vocab_size, embedding_dim = embed_size)
		#activations
		self.softmax = nn.Softmax(dim = 1)

	def forward(self, features, captions):
		batch_size = features.size(0)
		#initialise the hidden and cell states to zeros
		hidden_state = torch.zeros((batch_size, self.hidden_size))
		cell_state = torch.zeros((batch_size, self.hidden_size))

		#define the output tensor placeholder
		outputs = torch.empty((batch_size, captions_size(1), self.vocab_size))

		#embed the captions 
		captions_embed = self.embed(captions)

		#pass the captions word by word
		for t in range(captions.size(1)):
			# for the first time step the input is the feature vector
			if t == 0:
				hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

			else:
				hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

			#output of the attention mechanism
			out = self.fc_out(hidden_state)

			#build the output tensor
			outputs[:, t, :] = out

		return output
