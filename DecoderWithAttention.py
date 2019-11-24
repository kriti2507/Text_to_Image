# class DecoderWithAttention(nn.Module):
# 	"""
# 	DecoderWithAttention
# 	"""
# 	def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim = 2048, dropout = 0.25):
# 		"""
# 		parameters: attention_dim : size of attention network
# 					embed_dim : embedding size
# 					decoder_dim: size of decoder's RNN
# 					vocab_size: size of vocabulary
# 					encoder_dim: feature size of encoded images
# 					dropout: dropout
# 		"""
# 		super(DecoderWithAttention, self).__init__()
# 		self.attention_dim = attention_dim
# 		self.embed_dim = embed_dim
# 		self.decoder_dim = decoder_dim
# 		self.vocab_size = vocab_size
# 		self.encoder_dim = encoder_dim
# 		self.droput = droput

# 		self.attention = Attention(encoder_dim, decoder_dim, attention_dim) #attention network

# 	def init_weights(self):
# 		"""
# 		Initializes some parameters with values from the uniform distribution, for easier convergence.
# 		"""
# 		self.embedding.weights.data.uniform_(-0.1, 0.1)
# 		self.fc.bias.data.fill_(0)
# 		self.fc.weight.data.uniform_(-0.1, 0.1)

# 	def load_pretrained_embeddings(self, embeddings):
# 		"""
# 		Loads embedding layer with pre-trained embeddings.
# 		param embeddings: pre-trained embeddings
# 		"""
# 		self.embedding.weight = nn.Parameter(embeddings)

# 	def fine_tune_embeddings(self, fine_tune = True):
# 		for p in self.embedding.parameters():
# 			p.requires_grad = fine_tune

# 	def init_hidden_state(self, encoder_out):
# 		"""
# 		Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
# 		param : encoder_out : encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
# 		return : hidden state, cell state
# 		"""

# 		mean_encoder_out = encoder_out.mean(dim = 1)
# 		h = self.init_h(mean_encoder_out)
# 		c = self.init_c(mean_encoder_out)
# 		return h, c

# 	def forward(self, encoder_out, encoded_captions, caption_lengths):
# 		"""
# 		Forward propagation

# 		param encoder_out : encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
# 			encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
# 			caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
# 		return scores for vocabulary, sorted encoded captionsscores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices, decode lengths, weights, sort indices
# 		"""		
# 		batch_size = encoder_out.size(0)
# 		encoder_dim = encoder_out.size(-1)
# 		vocab_size = self.vocab_size
# 		#Flatten image
# 		encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
