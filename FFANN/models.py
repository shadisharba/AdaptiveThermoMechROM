import torch
from torch import nn
import copy


class BaseModule(nn.Module):
    """Represents a `Base Module` that contains the basic functionality of an artificial neural network (ANN).

    All modules should inherit from :class:`mlprum.models.BaseModule` and override :meth:`mlprum.models.BaseModule.forward`.
    The Base Module itself inherits from :class:`torch.nn.Module`. See the `PyTorch` documentation for further information.
    """    
    def __init__(self):
        """Constructor of the class. Initialize the Base Module.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`mlprum.models.BaseModule` should not be instantiated itself, but only its subclasses.
        """        
        super().__init__()
        self.device = 'cpu'

    def forward(*args):
        """Forward propagation of the ANN. Subclasses must override this method.

        :raises NotImplementedError: If the method is not overriden by a subclass.
        """        
        raise NotImplementedError('subclasses must override forward()!')  

    def training_step(self, dataloader, loss_fn, optimizer):
        """Single training step that performs the forward propagation of an entire batch,
        the training loss calculation and a subsequent optimization step.

        A training epoch must contain one call to this method.

        Example:
            >>> train_loss = module.training_step(train_loader, loss_fn, optimizer)

        :param dataloader: Dataloader with training data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for the model training
        :type loss_fn: method
        :param optimizer: Optimizer for model training
        :type optimizer: :class:`torch.optim.Optimizer`
        :return: Training loss
        :rtype: float
        """        
        self.train() # enable training mode
        cumulative_loss = 0
        samples = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Loss calculation
            y_pred = self(x)
            loss = loss_fn(y_pred, y)
            cumulative_loss += loss.item() * x.size(0)
            samples += x.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = cumulative_loss / samples
        return average_loss

    def loss_calculation(self, dataloader, loss_fns):
        """Perform the forward propagation of an entire batch from a given `dataloader`
        and the subsequent loss calculation for one or multiple loss functions in `loss_fns`.

        Example:
            >>> val_loss = module.loss_calculation(val_loader, loss_fn)

        :param dataloader: Dataloader with validation data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for model training
        :type loss_fn: method or list of methods
        :return: Validation loss
        :rtype: float
        """        
        self.eval() # disable training mode
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]
        cumulative_loss = torch.zeros(len(loss_fns))
        samples = 0
        with torch.no_grad(): # disable gradient calculation
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                samples += x.size(0)
                for i, loss_fn in enumerate(loss_fns):
                    loss = loss_fn(y_pred, y)
                    cumulative_loss[i] += loss.item() * x.size(0)
        average_loss = cumulative_loss / samples
        if torch.numel(average_loss) == 1:
            average_loss = average_loss[0]
        return average_loss

    def parameter_count(self):
        """Get the number of learnable parameters, that are contained in the model.

        :return: Number of learnable parameters
        :rtype: int
        """        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def reduce(loss, reduction='mean'):
        """Perform a reduction step over all datasets to transform a loss function to a cost function.

        A loss function is evaluated element-wise for a dataset.
        However, a cost function should return a single value for the dataset.
        Typically `mean` reduction is used.

        :param loss: Tensor that contains the element-wise loss for a dataset
        :type loss: :class:`torch.Tensor`
        :param reduction: ('mean'|'sum'), defaults to 'mean'
        :type reduction: str, optional
        :return: Reduced loss
        :rtype: float
        """        
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

    @staticmethod
    def unsqueeze(output, target):
        """Ensure that the tensors :code:`output` and :code:`target` have a shape of the form :code:`(N, features)`.

        When a loss function is called with a single data point, the tensor shape is :code:`(features)` and hence does not fit.
        This method expands the dimensions if needed.

        TODO: check for bugs

        :param output: Model output
        :type output: :class:`torch.Tensor`
        :param target: Target data
        :type target: :class:`torch.Tensor`
        :return: Tuple (output, target)
        :rtype: tuple
        """        
        if output.dim() == 1:
            output = torch.unsqueeze(output, 0)
        if target.dim() == 1:
            target = torch.unsqueeze(target, 0)
        return output, target

    def to(self, device, *args, **kwargs):
        """Transfers a model to another device, e.g. to a GPU.

        This method overrides the PyTorch built-in method :code:`model.to(...)`.

        Example:
            >>> # Transfer the model to a GPU
            >>> module.to('cuda:0')

        :param device: Identifier of the device, e.g. :code:`'cpu'`, :code:`'cuda:0'`, :code:`'cuda:1'`, ...
        :type device: str
        :return: The model itself
        :rtype: :class:`mlprum.models.BaseModule`
        """        
        self.device = device
        return super().to(device, *args, **kwargs)

    @property
    def gpu(self):
        """Property, that indicates, whether the model is on a GPU, i.e. not on the CPU.

        :return: True, iff the module is not on the CPU
        :rtype: bool
        """        
        return self.device != 'cpu'


class FFModule(BaseModule):
    """General feedforward neural network.

    It consists of an input layer with :code:`in_dim` neurons, multiple hidden layers with the
    neuron counts from the list :code:`neurons`, activation functions from the list :code:`activations`
    and an output layer with :code:`out_dim` neurons with the last element from :code:`activations` as activation function.

    Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], 35, [nn.SELU(), nn.SELU(), nn.Softplus()])
        >>> print(model) # print model summary
        ...
        >>> # model training based on data (x,y)...
        >>> y_pred = model(x) # model prediction

    """    
    def __init__(self, in_dim, neurons, activations, out_activation, out_dim, prepare_fn=None):
        """Constructor of the class. Initialize a feedforward neural network with given neuron counts and activation functions.
        
        Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], [nn.SELU(), nn.SELU()], nn.Softplus(), 35)
        
        :param in_dim: Number of neurons in the input layer, i.e. number of input features
        :type in_dim: int
        :param neurons: List of neuron counts in the hidden layers
        :type neurons: list
        :param out_dim: Number of neurons in the output layer, i.e. number of output features
        :type out_dim: int
        :param activations: List of activation functions for the hidden layers and the output layer
        :type activations: list
        :param prepare_fn: Method for preprocessing that is called before the input layer, defaults to None
        :type prepare_fn: method, optional
        """        
        super().__init__()
        self.neurons = [in_dim, *neurons]
        self.activations = activations
        self.hidden_layers = []
        self.prepare_fn = prepare_fn

        # Create the hidden layers
        for i, activation in enumerate(self.activations):
            self.hidden_layers.append(nn.Linear(self.neurons[i], self.neurons[i+1]))
            self.hidden_layers.append(activation)
        self.hidden = nn.Sequential(*self.hidden_layers)

        # Create the output layer
        self.output = nn.Sequential(nn.Linear(self.neurons[-1], out_dim), out_activation)
        
        # Initialization for SELU activation function
        """
        for param in self.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
        """

    def forward(self, x):
        """Forward propagation of the deterministic model.
        
        Do not call the method `forward` itself, but call the module directly.

        Example:
            >>> y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """ 
        z = self.intermediate(x)
        pred = self.output(z)
        return pred

    def intermediate(self, x):
        if self.prepare_fn is not None:
            x = self.prepare_fn(x)
        pred = self.hidden(x)
        return pred


class RectFFModule(FFModule):
    """General feedforward neural network with a rectangular shape for the hidden layers.

    Example:
        >>> # Create a feedforward ANN with 4 hidden layers (each 50 neurons)
        >>> model = RectFFModel(8, 50, 4, nn.SELU(), nn.Softplus(), 35) # create module
        >>> print(model) # print model summary
        ...
        >>> # model training based on data (x,y)...
        >>> y_pred = model(x) # model prediction

    """    
    def __init__(self, in_dim, neuron_count, layer_count, activation, out_activation, out_dim, prepare_fn=None):
        """Constructor of the class. Initialize the neural network with given neuron count, layer count and activation functions.

        Example:
        >>> # Create feedforward ANN with 4 hidden layers (50 neurons each)
        >>> module = RectFFModel(8, 50, 4, nn.SELU(), nn.Softplus(), 35)
              
        :param in_dim: Number of neurons in the input layer, i.e. number of input features
        :type in_dim: int
        :param neuron_count: Number of neurons in each hidden layer
        :type neuron_count: int
        :param layer_count: Number of hidden layers
        :type layer_count: int
        :param activation: Activation function for the hidden layers
        :type activation: method
        :param out_activation: Activation function for the output layer
        :type out_activation: method
        :param out_dim: Number of neurons in the input layer, i.e. number of input features
        :type out_dim: int
        :param prepare_fn: Method for preprocessing that is called before the input layer, defaults to None
        :type prepare_fn: method, optional
        """        
        neurons, activations = [], []
        for _ in range(layer_count):
            neurons.append(neuron_count)
            activations.append(copy.deepcopy(activation))
        super().__init__(in_dim, neurons, activations, out_activation, out_dim, prepare_fn)


class TransferModule(BaseModule):
    """General feedforward neural network.

    It consists of an input layer with :code:`in_dim` neurons, multiple hidden layers with the
    neuron counts from the list :code:`neurons`, activation functions from the list :code:`activations`
    and an output layer with :code:`out_dim` neurons with the last element from :code:`activations` as activation function.

    Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], 35, [nn.SELU(), nn.SELU(), nn.Softplus()])
        >>> print(model) # print model summary
        ...
        >>> # model training based on data (x,y)...
        >>> y_pred = model(x) # model prediction

    """    
    def __init__(self, trained_module, new_module1, new_module2):
        """Constructor of the class. Initialize a feedforward neural network with given neuron counts and activation functions.
        
        Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], [nn.SELU(), nn.SELU()], nn.Softplus(), 35)
        
        :param in_dim: Number of neurons in the input layer, i.e. number of input features
        :type in_dim: int
        :param neurons: List of neuron counts in the hidden layers
        :type neurons: list
        :param out_dim: Number of neurons in the output layer, i.e. number of output features
        :type out_dim: int
        :param activations: List of activation functions for the hidden layers and the output layer
        :type activations: list
        :param prepare_fn: Method for preprocessing that is called before the input layer, defaults to None
        :type prepare_fn: method, optional
        """        
        super().__init__()
        self.trained_module = trained_module
        self.new_module1 = new_module1
        self.new_module2 = new_module2

        # Fix the parameters of the trained module
        for param in trained_module.parameters():
            param.requires_grad = False


    def forward(self, x):
        """Forward propagation of the combined modules for transfer learning.
        
        Do not call the method `forward` itself, but call the module directly.

        Example:
            >>> y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """ 
        z = self.trained_module(x)
        pred = torch.hstack((self.new_module1(z), self.new_module2(z)))
        return pred


class SigmaSoftplus(nn.Module):
    """Use the softplus activation for the :math:`\\sigma` values in the output layer.
    Acts as an identity function for the :math:`\\mu` values in the output layer.

    :param nn: [description]
    :type nn: [type]
    """
    def __init__(self, beta=1, threshold=20):
        super(SigmaSoftplus, self).__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        mu, sigma = torch.chunk(x, 2, dim=1)
        sigma_softplus = self.softplus(sigma)
        output = torch.hstack((mu, sigma_softplus))
        return output


class MSModel(BaseModule):
    """Deterministic model to predict the response of a microstructure.

    The input features are given by the material parameters of two materials in a MMC and the volume fraction of the microstructure.
    The effective material parameters of the MMC that are obtained by a simulation with `Combo FANS`,
    are used as target data for the output features.

    This class acts as a base class for possible realizations. It only provides the class structure and the loss functions for training.
    """    
    def __init__(self):
        """Constructor of the class. Initialize the deterministic model.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`mlprum.models.MSModel` should not be instantiated itself, but only its subclasses.
        """        
        super().__init__()

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example:
            >>> y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        :raises NotImplementedError: If the method is not overriden by a subclass.
        """        
        raise NotImplementedError('subclasses must override the method `forward`!')

    @staticmethod
    def stiffness_loss(output, target, reduction='mean'):
        """Loss function for the output features, that describe the effective stiffness tensor.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_L = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(\\underline{\\hat{L}}_{\\mathrm{pred},i}, \\underline{\\hat{L}}_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        Should be used to train the stiffness module together with the dataset wrapper :class:`mlprum.data.StiffnessDataset`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        L, L_pred = (target, output[:,:21]) if target.size(1) == 21 else (target[:,:21], output[:,:21])
        loss = (torch.linalg.norm(L - L_pred, dim=1) / torch.linalg.norm(L, dim=1))**2
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def thermal_exp_loss(output, target, reduction='mean'):
        """Loss function for the output features, that describe the effective coefficient of thermal expansion tensor.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_A = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(\\underline{\\hat{A}}_{\\mathrm{pred},i}, \\underline{\\hat{A}}_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        Should be used to train the thermoelastic module together with the dataset wrapper :class:`mlprum.data.ThermoelDataset`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        alpha, alpha_pred = (target, output[:,:6]) if target.size(1) == 6 else (target[:,21:27], output[:,21:27])
        loss = (torch.linalg.norm(alpha - alpha_pred, dim=1) / torch.linalg.norm(alpha, dim=1))**2
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def conductivity_loss(output, target, reduction='mean'):
        """Loss function for the output features, that describe the effective conductivity tensor.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_K = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(\\underline{\\hat{K}}_{\\mathrm{pred},i}, \\underline{\\hat{K}}_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 6:
            kappa, kappa_pred = target, output[:,:6]
        elif target.size(1) == 8:
            kappa, kappa_pred = target[:,:6], output[:,:6]
        else:
            kappa, kappa_pred = target[:,27:33], output[:,27:33]
        loss = (torch.linalg.norm(kappa - kappa_pred, dim=1) / torch.linalg.norm(kappa, dim=1))**2
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def heat_capacity_loss(output, target, reduction='mean'):
        """Loss function for the output features, that describe the effective heat capacity.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_c = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(c_{\\mathrm{pred},i}, c_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 1:
            c, c_pred = target, output[:,0]
        elif target.size(1) == 8:
            c, c_pred = target[:,6], output[:,6]
        else:
            c, c_pred = target[:,33], output[:,33]
        c, c_pred = c.view(-1,1), c_pred.view(-1,1)
        loss = torch.square((c - c_pred) / c)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def density_loss(output, target, reduction='mean'):
        """Loss function for the output features, that describe the effective density.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\rho = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(\\rho_{\\mathrm{pred},i}, \\rho_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 1:
            rho, rho_pred = target, output[:,0]
        elif target.size(1) == 8:
            rho, rho_pred = target[:,7], output[:,7]
        else:
            rho, rho_pred = target[:,34], output[:,34]
        rho, rho_pred = rho.view(-1,1), rho_pred.view(-1,1)
        loss = torch.square((rho - rho_pred) / rho)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def mech_loss(output, target, reduction='mean'):
        """Loss function for the mechanical output features.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\mathrm{mech} = \\mathcal{L}_L + \\mathcal{L}_A.

        Should be used to train the mechanical module together with the dataset wrapper :class:`mlprum.data.MechDataset`.

        See :meth:`mlprum.models.MSModel.stiffness_loss` and :meth:`mlprum.models.MSModel.thermal_exp_loss`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """
        output, target = __class__.unsqueeze(output, target)
        L, L_pred = target[:,:21], output[:,:21]
        alpha, alpha_pred = target[:,21:27], output[:,21:27]
        loss = __class__.stiffness_loss(L_pred, L, reduction) + \
               __class__.thermal_exp_loss(alpha_pred, alpha, reduction)
        return loss

    @staticmethod
    def remaining_loss(output, target, reduction='mean'):
        """Loss function for the non-mechanical output features.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\mathrm{remaining} = \\mathcal{L}_K + \\mathcal{L}_c + \\mathcal{L}_\\rho.

        Should be used to train the remaining module together with the dataset wrapper :class:`mlprum.data.RemainingDataset`.

        See :meth:`mlprum.models.MSModel.conductivity_loss`, :meth:`mlprum.models.MSModel.heat_capacity_loss`
        and :meth:`mlprum.models.MSModel.density_loss`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """
        output, target = __class__.unsqueeze(output, target)
        K, K_pred = (target[:,:6], output[:,:6]) if target.size(1) == 8 else (target[:,27:33], output[:,27:33])
        c, c_pred = (target[:,6], output[:,6]) if target.size(1) == 8 else (target[:,33], output[:,33])
        c, c_pred = c.view(-1,1), c_pred.view(-1,1)
        rho, rho_pred = (target[:,7], output[:,7]) if target.size(1) == 8 else (target[:,34], output[:,34])
        rho, rho_pred = rho.view(-1,1), rho_pred.view(-1,1)
        loss = __class__.conductivity_loss(K_pred, K, reduction) + \
               __class__.heat_capacity_loss(c_pred, c, reduction) + \
               __class__.density_loss(rho_pred, rho, reduction)
        return loss

    @staticmethod
    def therm_loss(output, target, reduction='mean'):
        """Loss function for the thermal output features.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\mathrm{therm} = \\mathcal{L}_K.

        Should be used to train the thermal module together with the dataset wrapper :class:`mlprum.data.ThermDataset`.

        See :meth:`mlprum.models.MSModel.conductivity_loss`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """
        output, target = __class__.unsqueeze(output, target)
        loss = __class__.conductivity_loss(output, target, reduction)
        return loss

    @staticmethod
    def deterministic_loss(output, target, reduction='mean'):
        """Loss function for the non-mechanical output features.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\mathrm{remaining} = \\mathcal{L}_K + \\mathcal{L}_c + \\mathcal{L}_\\rho

        Is only included for completeness.

        See :meth:`mlprum.models.MSModel.conductivity_loss`, :meth:`mlprum.models.MSModel.heat_capacity_loss`
        and :meth:`mlprum.models.MSModel.density_loss`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """
        output, target = __class__.unsqueeze(output, target)
        c, c_pred = (target[:,0], output[:,0]) if target.size(1) == 8 else (target[:,33], output[:,33])
        c, c_pred = c.view(-1,1), c_pred.view(-1,1)
        rho, rho_pred = (target[:,1], output[:,1]) if target.size(1) == 8 else (target[:,34], output[:,34])
        rho, rho_pred = rho.view(-1,1), rho_pred.view(-1,1)
        loss = __class__.heat_capacity_loss(c_pred, c, reduction) + \
               __class__.density_loss(rho_pred, rho, reduction)
        return loss

    @staticmethod
    def total_loss(output, target, reduction='mean'):
        """Loss function for all output features.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_\\mathrm{total} = \\mathcal{L}_L + \\mathcal{L}_A + \\mathcal{L}_K + \\mathcal{L}_c + \\mathcal{L}_\\rho.

        Should be used to train the overall module together with the dataset :class:`mlprum.data.Dataset`.

        See :meth:`mlprum.models.MSModel.stiffness_loss`, :meth:`mlprum.models.MSModel.thermal_exp_loss`,
        :meth:`mlprum.models.MSModel.conductivity_loss`, :meth:`mlprum.models.MSModel.heat_capacity_loss`
        and :meth:`mlprum.models.MSModel.density_loss`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """
        output, target = __class__.unsqueeze(output, target)
        loss = __class__.mech_loss(output, target, reduction) + \
               __class__.remaining_loss(output, target, reduction)
        return loss


class PMSModel(BaseModule):
    """Probabilistic model to predict the response of a microstructure.

    The input features are given by the material parameters of two materials in a MMC and the volume fraction of the microstructure.
    The effective material parameters of the MMC that are obtained by a simulation with `Combo FANS`.
    The output features are given by the mean and the standard deviation of the effective material parameters.
    Thus the probabilistic model (:class:`mlprum.models.PMSModel`) has twice as many output features as the deterministic model (:class:`mlprum.models.MSModel`).

    This class acts as a base class for possible realizations. It only provides the class structure and the loss functions for training.
    """    
    def __init__(self):
        """Constructor of the class. Initialize the probabilistic model.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`mlprum.models.PMSModel` should not be instantiated itself, but only its subclasses.
        """        
        super().__init__()

    def forward(self, x):
        """Forward propagation of the probabilistic model.
        
        Do not call the function `forward` itself, but call the model directly.

        Example:
            >>> y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        :raises NotImplementedError: If the method is not overriden by a subclass.
        """        
        raise NotImplementedError('subclasses must override the method `forward`!')  

    @staticmethod
    def stiffness_loss(output, target, reduction='mean'):
        """Probabilistic loss function for the output features, that describe the effective stiffness tensor and its uncertainty.
        The loss function quantifies the deviation of the model prediction `output` from the ground truth `target`.
        If `reduction` is set to 'mean' (default case), this method acts as a cost function that returns
        the average over all element-wise losses.

        The mathematical definition of the used cost function is

        .. math::
            \\mathcal{L}_L = \\frac{1}{N} \\sum_{i=1}^N
                l_\\mathrm{NMSE}(\\underline{\\hat{L}}_{\\mathrm{pred},i}, \\underline{\\hat{L}}_{\\mathrm{eff},i})

        with the normalized mean squared error (NMSE) loss given by

        .. math::
            \\newcommand{\\normg}[1]{\\left\\lVert#1\\right\\rVert}
            l_\\mathrm{NMSE}(\\underline{y}_\\mathrm{pred}, \\underline{y}_\\mathrm{eff}) =
                \\frac{\\normg{\\underline{y}_\\mathrm{pred} - \\underline{y}_\\mathrm{eff}}^2_2}{\\normg{\\underline{y}_\\mathrm{eff}}^2_2}.

        Should be used to train the stiffness module together with the dataset wrapper :class:`mlprum.data.StiffnessDataset`.

        :param output: Model predictions
        :type output: :class:`torch.Tensor`
        :param target: Ground truth, i.e. reference data
        :type target: :class:`torch.Tensor`
        :param reduction: Reduction mode ('mean', 'sum', other: no reduction), defaults to 'mean'
        :type reduction: str, optional
        :return: Element-wise loss or cost (if `reduction` is 'mean')
        :rtype: :class:`torch.Tensor`
        """        
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 21:
            L, L_mu, L_sigma = target, output[:,:21], output[:,21:42]
        elif target.size(1) == 27:
            L, L_mu, L_sigma = target[:,:21], output[:,:21], output[:,27:48]
        else:
            L, L_mu, L_sigma = target[:,:21], output[:,:21], output[:,35:56]
        loss = __class__.NLLLoss(L_mu, L_sigma, L)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def thermal_exp_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 6:
            alpha, alpha_mu, alpha_sigma = target, output[:,:6], output[:,6:12]
        elif target.size(1) == 27:
            alpha, alpha_mu, alpha_sigma = target[:,21:27], output[:,21:27], output[:,48:56]
        else:
            alpha, alpha_mu, alpha_sigma = target[:,21:27], output[:,21:27], output[:,56:62]
        loss = __class__.NLLLoss(alpha_mu, alpha_sigma, alpha)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def conductivity_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 6:
            kappa, kappa_mu, kappa_sigma = target, output[:,:6], output[:,6:12]
        elif target.size(1) == 8:
            kappa, kappa_mu, kappa_sigma = target[:,:6], output[:,:6], output[:,8:14]
        else:
            kappa, kappa_mu, kappa_sigma = target[:,27:33], output[:,27:33], output[:,62:68]
        loss = __class__.NLLLoss(kappa_mu, kappa_sigma, kappa)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def heat_capacity_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 1:
            c, c_mu, c_sigma = target, output[:,0], output[:,1]
        elif target.size(1) == 8:
            c, c_mu, c_sigma =  target[:,6], output[:,6], output[:,14]
        else:
            c, c_mu, c_sigma =  target[:,33], output[:,33], output[:,68]
        c, c_mu, c_sigma = c.view(-1,1), c_mu.view(-1,1), c_sigma.view(-1,1)
        loss = __class__.NLLLoss(c_mu, c_sigma, c)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def density_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        if target.size(1) == 1:
            rho, rho_mu, rho_sigma = target, output[:,0], output[:,1]
        elif target.size(1) == 8:
            rho, rho_mu, rho_sigma =  target[:,7], output[:,7], output[:,15]
        else:
            rho, rho_mu, rho_sigma =  target[:,34], output[:,34], output[:,69]
        rho, rho_mu, rho_sigma = rho.view(-1,1), rho_mu.view(-1,1), rho_sigma.view(-1,1)
        loss = __class__.NLLLoss(rho_mu, rho_sigma, rho)
        loss = __class__.reduce(loss, reduction)
        return loss

    @staticmethod
    def mech_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        L, L_mu, L_sigma = (target[:,:21], output[:,:21], output[:,27:48]) if target.size(1) == 27 else (target[:,:21], output[:,:21], output[:,35:56])
        alpha, alpha_mu, alpha_sigma = (target[:,21:27], output[:,21:27], output[:,48:54]) if target.size(1) == 27 else (target[:,21:27], output[:,21:27], output[:,56:62])
        L_pred = torch.hstack((L_mu, L_sigma))
        alpha_pred = torch.hstack((alpha_mu, alpha_sigma))
        loss = __class__.stiffness_loss(L_pred, L, reduction) + \
               __class__.thermal_exp_loss(alpha_pred, alpha, reduction)
        return loss

    @staticmethod
    def remaining_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        K, K_mu, K_sigma = (target[:,:6], output[:,:6], output[:,8:14]) if target.size(1) == 8 else (target[:,27:33], output[:,27:33], output[:,62:68])
        c, c_mu, c_sigma = (target[:,6], output[:,6], output[:,14]) if target.size(1) == 8 else (target[:,33], output[:,33], output[:,68])
        c, c_mu, c_sigma = c.view(-1,1), c_mu.view(-1,1), c_sigma.view(-1,1)
        rho, rho_mu, rho_sigma = (target[:,7], output[:,7], output[:,15]) if target.size(1) == 8 else (target[:,34], output[:,34], output[:,69])
        rho, rho_mu, rho_sigma = rho.view(-1,1), rho_mu.view(-1,1), rho_sigma.view(-1,1)
        K_pred = torch.hstack((K_mu, K_sigma))
        c_pred = torch.hstack((c_mu, c_sigma))
        rho_pred = torch.hstack((rho_mu, rho_sigma))
        loss = __class__.conductivity_loss(K_pred, K, reduction) + \
               __class__.heat_capacity_loss(c_pred, c, reduction) + \
               __class__.density_loss(rho_pred, rho, reduction)
        return loss

    @staticmethod
    def therm_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        loss = __class__.conductivity_loss(output, target, reduction)
        return loss

    @staticmethod
    def deterministic_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        c, c_mu, c_sigma = (target[:,0], output[:,0], output[:,2]) if target.size(1) == 2 else (target[:,33], output[:,33], output[:,68])
        rho, rho_mu, rho_sigma = (target[:,1], output[:,1], output[:,3]) if target.size(1) == 2 else (target[:,34], output[:,34], output[:,69])
        c_pred = torch.hstack((c_mu, c_sigma))
        rho_pred = torch.hstack((rho_mu, rho_sigma))
        loss = __class__.heat_capacity_loss(c_pred, c, reduction) + \
               __class__.density_loss(rho_pred, rho, reduction)
        return loss

    @staticmethod
    def total_loss(output, target, reduction='mean'):
        output, target = __class__.unsqueeze(output, target)
        loss = __class__.mech_loss(output, target, reduction) + \
               __class__.remaining_loss(output, target, reduction)
        return loss
        
    @staticmethod
    def NLLLoss(output_mu, output_sigma, target):
        """Negative log-likelihood loss (NLL) for the PMS model

        .. math::

            \\mathcal{L}(y_{pred}, y) = \\mathrm{log}(\\sigma) + \\frac{y - \\mu}{2\\sigma^2}, \\quad y_{pred} = (\\mu, \\sigma)


        Example::

            # let (x,y) be the ground truth
            y_pred = model(x) # predict mu and sigma
            loss = PMSModel.NLLLoss(y_pred, y) # calculate NLL loss

        :param output_mu: mu prediction/output of the ANN
        :type output_mu: Tensor
        :param output_sigma: sigma prediction/output of the ANN
        :type output_sigma: Tensor
        :param target: expected ground truth
        :type target: Tensor
        :return: NLL loss for given output and target
        :rtype: Tensor
        """
        loss = torch.sum(torch.log(output_sigma) + torch.square(target - output_mu) / (2*torch.square(output_sigma)), dim=1)
        #if torch.all(output_sigma == 0):
        #    loss = 0 # NLL Loss for deterministic outputs
        return loss


class MS1Model(MSModel):
    """Deterministic model to predict the response of a microstructure
    
    Variant 1: overall model to predict all effective properties
    """    
    def __init__(self, overall_module):
        """Initialize the neural network

        Example::

            overall_module = ... # input dim: 8, output dim: 32
            model = MS1Model(overall_module)
        
        """        
        super().__init__()
        self.overall_module = overall_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        pred = self.overall_module(x)
        return pred


class PMS1Model(PMSModel):
    """Probabilistic model to predict the response of a microstructure
    
    Variant 1: overall model to predict all effective properties
    """    
    def __init__(self, overall_module):
        """Initialize the neural network

        Example::

            overall_module = ... # input dim: 8, output dim: 64
            model = PMS1Model(overall_module)
        
        """        
        super().__init__()
        self.overall_module = overall_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        pred = self.overall_module(x)
        return pred


class MS2Model(MSModel):
    """Deterministic model to predict the response of a microstructure
    
    Variant 2: mechanical module and thermal module (including heat capacity)
    """    
    def __init__(self, mech_module, remaining_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 5, output dim: 24
            remaining_module = ... # input dim: 4, output dim: 8
            model = MS2Model(mech_module, remaining_module)
        
        """        
        super().__init__()
        self.mech_module = mech_module
        self.remaining_module = remaining_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data (8 values)
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        mech_x = x[:,[0,1,2,3,7]]
        remaining_x = x[:,[4,5,6,7]]
        mech_pred = self.mech_module(mech_x)
        remaining_pred = self.remaining_module(remaining_x)
        pred = torch.hstack((mech_pred, remaining_pred))
        return pred


class PMS2Model(MSModel):
    """Probabilistic model to predict the response of a microstructure
    
    Variant 2: mechanical module and thermal module (including heat capacity)
    """    
    def __init__(self, mech_module, remaining_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 5, output dim: 48
            remaining_module = ... # input dim: 4, output dim: 16
            model = PMS3Model(mech_module, remaining_module)
        
        """        
        super().__init__()
        self.mech_module = mech_module
        self.remaining_module = remaining_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        mech_x = x[:,[0,1,2,3,7]]
        remaining_x = x[:,[4,5,6,7]]
        mech_pred = self.mech_module(mech_x)
        remaining_pred = self.remaining_module(remaining_x)
        mech_mu, mech_sigma = torch.chunk(mech_pred, 2, dim=1)
        remaining_mu, remaining_sigma = torch.chunk(remaining_pred, 2, dim=1)
        pred = torch.hstack((mech_mu, remaining_mu, mech_sigma, remaining_sigma))
        return pred


class MS3Model(MSModel):
    """Deterministic model to predict the response of a microstructure
    
    Variant 3: mechanical module, thermal module and deterministic heat capacity
    """    
    def __init__(self, mech_module, therm_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 5, output dim: 24
            therm_module = ... # input dim: 2, output dim: 6
            model = MS3Model(mech_module, therm_module)
        
        """        
        super().__init__()
        self.mech_module = mech_module
        self.therm_module = therm_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        mech_x = x[:,[0,1,2,3,7]]
        therm_x = x[:,[4,7]]
        c, rho, f1 = x[:,5], x[:,6], x[:,7]
        mech_pred = self.mech_module(mech_x)
        therm_pred = self.therm_module(therm_x)
        rho_eff = (1-f1) + f1*rho
        c_eff = ((1-f1) + f1*rho*c) / rho_eff
        pred = torch.hstack((mech_pred, therm_pred, c_eff.view(-1,1), rho_eff.view(-1,1)))
        return pred


class PMS3Model(MSModel):
    """Probabilistic model to predict the response of a microstructure
    
    Variant 3: mechanical module, thermal module and deterministic heat capacity
    """    
    def __init__(self, mech_module, therm_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 5, output dim: 48
            therm_module = ... # input dim: 2, output dim: 12
            model = PMS3Model(mech_module, therm_module)
        
        """        
        super().__init__()
        self.mech_module = mech_module
        self.therm_module = therm_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        mech_x = x[:,[0,1,2,3,7]]
        therm_x = x[:,[4,7]]
        c, rho, f1 = x[:,5], x[:,6], x[:,7]
        mech_pred = self.mech_module(mech_x)
        therm_pred = self.therm_module(therm_x)
        rho_eff = (1-f1) + f1*rho
        c_eff = ((1-f1) + f1*rho*c) / rho_eff
        mech_mu, mech_sigma = torch.chunk(mech_pred, 2, dim=1)
        therm_mu, therm_sigma = torch.chunk(therm_pred, 2, dim=1)
        pred = torch.hstack((mech_mu, therm_mu, c_eff.view(-1,1), rho_eff.view(-1,1),
                             mech_sigma, therm_sigma, torch.zeros_like(c_eff).view(-1,1), torch.zeros_like(rho_eff).view(-1,1)))
        return pred


class MS4Model(MSModel):
    """Deterministic model to predict the response of a microstructure
    
    Variant 4: mechanical module, thermoelastic module, thermal module and deterministic heat capacity
    """    
    def __init__(self, stiffness_module, thermoel_module, therm_module):
        """Initialize the neural network

        Example::

            stiffness_module = ... # input dim: 4, output dim: 21
            thermoel_module = ... # input dim: 5, output dim: 3
            therm_module = ... # input dim: 2, output dim: 6
            model = MS3Model(stiffness_module, thermoel_module, therm_module)
        
        """        
        super().__init__()
        self.stiffness_module = stiffness_module
        self.thermoel_module = thermoel_module
        self.therm_module = therm_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        stiffness_x = x[:,[0,1,2,7]]
        thermoel_x = x[:,[0,1,2,3,7]]
        therm_x = x[:,[4,7]]
        c, rho, f1 = x[:,5], x[:,6], x[:,7]
        stiffness_pred = self.stiffness_module(stiffness_x)
        thermoel_pred = self.thermoel_module(thermoel_x)
        therm_pred = self.therm_module(therm_x)
        rho_eff = (1-f1) + f1*rho
        c_eff = ((1-f1) + f1*rho*c) / rho_eff
        pred = torch.hstack((stiffness_pred, thermoel_pred, therm_pred, c_eff.view(-1,1), rho_eff.view(-1,1)))
        return pred


class PMS4Model(MSModel):
    """Probabilistic model to predict the response of a microstructure
    
    Variant 4: stiffness module, thermoelastic module, thermal module and deterministic heat capacity
    """    
    def __init__(self, stiffness_module, thermoel_module, therm_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 4, output dim: 21
            thermoel_module = ... # input dim: 5, output dim: 3
            therm_module = ... # input dim: 2, output dim: 6
            model = MS3Model(mech_module, thermoel_module, therm_module)
        
        """        
        super().__init__()
        self.stiffness_module = stiffness_module
        self.thermoel_module = thermoel_module
        self.therm_module = therm_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        stiffness_x = x[:,[0,1,2,7]]
        thermoel_x = x[:,[0,1,2,3,7]]
        therm_x = x[:,[4,7]]
        c, rho, f1 = x[:,5], x[:,6], x[:,7]
        stiffness_pred = self.stiffness_module(stiffness_x)
        thermoel_pred = self.thermoel_module(thermoel_x)
        therm_pred = self.therm_module(therm_x)
        rho_eff = (1-f1) + f1*rho
        c_eff = ((1-f1) + f1*rho*c) / rho_eff
        stiffness_mu, stiffness_sigma = torch.chunk(stiffness_pred, 2, dim=1)
        thermoel_mu, thermoel_sigma = torch.chunk(thermoel_pred, 2, dim=1)
        therm_mu, therm_sigma = torch.chunk(therm_pred, 2, dim=1)
        pred = torch.hstack((stiffness_mu, thermoel_mu, therm_mu, c_eff.view(-1,1), rho_eff.view(-1,1),
                             stiffness_sigma, thermoel_sigma, therm_sigma, torch.zeros_like(c_eff).view(-1,1), torch.zeros_like(rho_eff).view(-1,1)))
        return pred


class MS5Model(MSModel):
    """Deterministic model to predict the response of a microstructure
    
    Variant 5: mechanical module (intermediate layers), stiffness module, thermoelastic module, thermal module and deterministic heat capacity
    """    
    def __init__(self, mech_module, mech_stiffness_module, mech_thermoel_module, therm_module):
        """Initialize the neural network

        Example::

            mech_module = ... # input dim: 5, output dim: z_dim
            mech_stiffness_module = ... # input dim: z_dim, output dim: 21
            mech_thermoel_module = ... # input dim: z_dim, output dim: 3
            therm_module = ... # input dim: 2, output dim: 6
            model = MS3Model(mech_module, mech_stiffness_module, mech_thermoel_module, therm_module)
        
        """        
        super().__init__()
        self.mech_module = mech_module
        self.mech_stiffness_module = mech_stiffness_module
        self.mech_thermoel_module = mech_thermoel_module
        self.therm_module = therm_module

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        #x = self.flatten(x)
        mech_x = x[:,[0,1,2,3,7]]
        therm_x = x[:,[4,7]]
        c, rho, f1 = x[:,5], x[:,6], x[:,7]
        z = self.mech_module.hidden(mech_x)
        stiffness_pred = self.mech_stiffness_module(z)
        thermoel_pred = self.mech_thermoel_module(z)
        therm_pred = self.therm_module(therm_x)
        rho_eff = (1-f1) + f1*rho
        c_eff = ((1-f1) + f1*rho*c) / rho_eff
        pred = torch.hstack((stiffness_pred, thermoel_pred, therm_pred, c_eff.view(-1,1), rho_eff.view(-1,1)))
        return pred


class DAModel(BaseModule):
    """Simple deterministic artificial neural network (dA model)

    It consists of an input layer with one neuron, 4 hidden layers (60, 40, 20, 10 neurons)
    with SELU activation function and an output layer with one neuron without activation function.

    Example::

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DAModel().to(device) # create model
        print(model) # print model summary
        # model training based on data (x,y)...
        y_pred = model(x) # model prediction

    """    
    def __init__(self):
        """Initialize the neural network

        Example::

            model = DAModel()
        
        """        
        super(DAModel, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(1, 100),
            nn.SELU(),
            nn.Linear(100, 80),
            nn.SELU(),
            nn.Linear(80, 40),
            nn.SELU(),
            nn.Linear(40, 10),
            nn.SELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        x = self.flatten(x)
        pred = self.dense_layers(x)
        return pred
        

class PAModel(BaseModule):
    """Simple propabilistic artificial neural network (pA model)

    It consists of an input layer with one neuron, 4 hidden layers (60, 40, 20, 10 neurons)
    with SELU activation function and an output layer with one neuron without activation function.

    Example::

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PAModel().to(device) # create model
        print(model) # print model summary
        # model training based on data (x,y)...
        y_pred = model(x) # model prediction
        # y_pred consists of a prediction for mu and sigma

    """    
    def __init__(self):
        """Initialize the neural network

        Example::

            model = PAModel()
        
        """        
        super(PAModel, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(1, 60),
            nn.SELU(),
            nn.Linear(60, 40),
            nn.SELU(),
            nn.Linear(40, 20),
            nn.SELU(),
            nn.Linear(20, 10),
            nn.SELU(),
            nn.Linear(10, 2) # yields mu and sigma
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Forward propagation of the neural network
        
        Do not call the function `forward` itself, but call the model directly.

        Example::

            y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """        
        x = self.flatten(x)
        pred = self.dense_layers(x) # yields mu and sigma
        mu = pred[:,0].view((-1,1)) # extract mu to avoid in-place operation
        sigma = self.softplus(pred[:,1]).view((-1,1)) # apply softmax for sigma
        pred = torch.cat((mu, sigma), dim=-1)
        return pred
        
    @staticmethod
    def NLLLoss(output, target):
        """Negative log-likelihood loss (NLL) for the pA model

        .. math::

            \mathcal{L}(y_{pred}, y) = \mathrm{log}(\sigma) + \\frac{y - \mu}{2\sigma^2}, \\quad y_{pred} = (\mu, \sigma)


        Example::

            # let (x,y) be the ground truth
            y_pred = model(x) # predict mu and sigma
            loss = PAModel.NLLLoss(y_pred, y) # calculate NLL loss

        :param output: prediction/output of the ANN
        :type output: Tensor
        :param target: expected ground truth
        :type target: Tensor
        :return: NLL loss for given output and target
        :rtype: Tensor
        """
        mu = torch.reshape(output[:,0], target.size())
        sigma = torch.reshape(output[:,1], target.size())
        loss = torch.mean(torch.log(sigma) + (target - mu)**2/(2*sigma**2))
        return loss
