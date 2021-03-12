import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False):
        super(Discriminator, self).__init__()
        self.nc = nc
        assert(imsize==32 or imsize==64 or imsize==128)
        self.imsize = imsize
        self.hflip = hflip

        SN = torch.nn.utils.spectral_norm
        IN = lambda x : nn.InstanceNorm2d(x)

        blocks = []
        if self.imsize==128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(in_channels=nc, out_channels=ndf//2, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(in_channels=ndf//2, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize==64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(in_channels=nc, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            SN(nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            #nn.BatchNorm2d(ndf * 4),
            IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            #nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, input, y=None):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)  # (BxN_samples)xC -> BxCxHxW

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        return self.main(input)


