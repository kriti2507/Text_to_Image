class Generator_one(nn.Module):
	def __init__(self):
		super(Generator_one,self).__init__()
		self.main = nn.Sequential(
			#input is z, going into convolution
			nn.ConvTranspose2d(nz, ng1 * 8, 4, bias = False),
			nn.BatchNorm2d(ng1 * 8),
			nn.ReLU(True),
			#State size (ng1 * 8) x 4 x 4
			nn.ConvTranspose2d(ng1 * 8, ng1 * 4, 4, bias = False),
			nn.BatchNorm2d(ng1 * 4),
			nn.ReLU(True),
			#State size (ng1 * 4) x 8 x 8 
			nn.ConvTranspose2d(ng1 * 4, ng1 * 2, 4, bias = False),
			nn.BatchNorm2d(ng1 * 2),
			nn.ReLU(True),
			#State size (ng1 * 2) x 16 x 16
			nn.ConvTranspose2d(ng1 * 2, ng1 , 4, bias = False),
			nn.BatchNorm2d(ng1),
			nn.ReLU(True),
			#State size (ng1) x 32 x 32
			nn.ConvTranspose2d(ng1, nc, 4, bias = False),
			nn.Tanh()
			#State size nc x 64 x 64
		)

	def forward(self,input):
		return self.main(input)
