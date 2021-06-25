from collections import OrderedDict
import numpy as np
from projet_etu import Module

class Sequential(Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._modules = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self,name,module):
        self._modules[name] = module

    def zero_grad(self):
        """
        applique le zero_grad a tout les modules dans _modules
        :return:None
        """
        for module in self._modules.values():
            module.zero_grad()

    def forward(self, X):
        """
        applique le forward pour chaque module avec comme entrée le resultat du forward précédent
        :param X:batch*input
        :return: batch*output ,le output du dernier module
        """
        input=X
        for modular in self._modules.values():
            input=modular.forward(input)
        self._forward=input
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """
        Applique l'update_parameters a tout les modules
        :param gradient_step: float
        :return: None
        """
        for module in self._modules.values():
            module.update_parameters(gradient_step)

    def backward_update_gradient(self, input, delta):
        """
        Applique le backward_update_gradient a tout les modules
        :param input:batch*input la premiere inpute
        :param delta: delta de la loss
        :return:
        """
        modules=list(self._modules.values())[::-1]
        next = modules[0]
        for module in modules[1:]:
            previous=module
            next.backward_update_gradient(previous._forward,delta)
            delta=next._delta
            next=previous
        next.backward_update_gradient(input, delta)

    def backward_delta(self, input, delta):
        """
        retroproge delta pour tout les modules
        :param input: batch*input ,les premieres entrées
        :param delta: batch*output , le dernier delta ,celui de la loss
        :return: delta ,on peut enchainé plusieur module sequentiel
        """
        modules = list(self._modules.values())[::-1]
        next = modules[0]
        for module in modules[1:]:
            previous = module
            delta = next.backward_delta(previous._forward, delta)
            next = previous
        self._delta = next.backward_delta(input,delta)
        return self._delta


    def __str__(self):
        string='Sequential :('
        for module in self._modules.values():
            string+=str(module)+", "
        string=string[:-2]+')'
        return string
