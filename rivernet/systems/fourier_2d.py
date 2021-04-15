import torch
from einops import repeat
from rivernet.modules import Module
from rivernet.modules.loss import LpLoss

from .base import System


@System.register('fourier_2d')
class Fourier2DSystem(System):
    def __init__(self, conv: Module, n_steps: int, model_path: str = None):
        super().__init__()
        self.conv = conv
        self.n_steps = n_steps
        self.l2_loss = LpLoss(size_average=True)

        if model_path:
            best_model_state = torch.load(model_path)
            self.conv.load_state_dict(best_model_state)

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze()

    def _learning_step(self, batch):
        xx, yy = batch
        B, X, Y, _ = xx.shape
        # xx.shape == [batch_size, x_dim, y_dim, in_channels]
        # yy.shape == [batch_size, x_dim, y_dim, out_channels]

        ticks = torch.linspace(0, 1, X).to(xx.device)
        grid_x = repeat(ticks, 'x -> b x y 1', b=B, y=Y)
        grid_y = repeat(ticks, 'y -> b x y 1', b=B, x=X)

        loss = 0
        for t in range(self.n_steps):
            y = yy[..., t: t+1]
            # y.shape == [batch_size, x_dim, y_dim, 1]

            im = self.conv(xx)
            loss += self.l2_loss(im.reshape(B, -1), y.reshape(B, -1))
            pred = im if t == 0 else torch.cat((pred, im), dim=-1)
            xx = torch.cat((xx[..., 1: -2], im, grid_x, grid_y), dim=-1)

        loss /= self.n_steps
        loss_full = self.l2_loss(pred.reshape(B, -1), yy.reshape(B, -1))

        return loss, loss_full

    def training_step(self, batch, batch_idx):
        loss, loss_full = self._learning_step(batch)
        self.log('train_loss', loss)
        self.log('train_loss_full', loss_full)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_full = self._learning_step(batch)
        self.log('valid_loss', loss)
        self.log('valid_loss_full', loss_full)

    def test_step(self, batch, batch_idx):
        loss, loss_full = self._learning_step(batch)
        self.log('test_loss', loss)
        self.log('test_loss_full', loss_full)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.0025, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=100, gamma=0.5)
        return [opt], [scheduler]
