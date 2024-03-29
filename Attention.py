class Attention(nn.Module):
	def __init__(self, encoder_dim, decoder_dim, attention_dim):
		super(Attention, self).__init__()
		self.encoder_dim = encoder_dim
		self.decoder_dim = decoder_dim
		self.attention_dim = attention_dim

	def forward(self, encoder_out, decoder_hidden):
		"""
		Forward Propagation
		parameters :encoder_out - encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
					decoder_hidden - previous decoder output, a tensor of dimension (batch_size, decoder_dim)

		return : Attention weighted encoding, weights
		"""

		att1 = nn.Linear(encoder_dim, attention_dim)(encoder_out) # (batch_size, num_pixels, attention_dim)
		att2 = nn.Linear(decoder_dim. attention_dim)(decoder_hidden)  # (batch_size, attention_dim)
		att = nn.Linear(attention_dim, 1)(nn.ReLU()(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
		alpha = nn.Softmax(dim = 1)(att) # (batch_size, num_pixels)
		attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)   # (batch_size, encoder_dim)
		
		return attention_weighted_encoding, alpha
