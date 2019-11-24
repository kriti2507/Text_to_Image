class Discriminator_one(nn.Module):
	def __init__(self):
		super(Discriminator_one, self).__init__()
		self.main = nn.Sequential(
			#Input is nc x 64 x 64
			nn.Conv2d(nc, nd1, 4, bias = False),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1) x 32 x 32
			nn.Conv2d(nd1, nd1 * 2, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 2),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 2) x 16 x 16
			nn.Conv2d(nd1 * 2, nd1 * 4, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 4),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 4) x 8 x 8
			nn.Conv2d(nd1 * 4, nd1 * 8, 4 , bias = False),
			nn.BatchNorm2d(nd1 * 8),
			nn.LeakyReLU(0.2, inplace = True),
			#State size (nd1 * 8) x 4 x 4
			nn.Conv2d(nd1 * 8, 1, 4, 1, 0, bias = False),
			nn.Sigmoid()
		)
	def forward(self, input):
		return self.main(input)
