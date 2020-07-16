import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from xnas.search_space.cell_based import *
from xnas.search_space.cell_based import _MixedOp
from torch.autograd import Variable


'''
Darts: highly copyed from https://github.com/khanrc/pt.darts
'''


class PCDartsCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.net = net
        self.device_ids = device_ids
        self.n_ops = len(self.net.basic_op_list)
        self.alpha = nn.Parameter(
            1e-3*torch.randn(self.net.all_edges, self.n_ops))
        self.beta = nn.Parameter(
            1e-3*torch.randn(self.net.all_edges))

        self.criterion = criterion

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        print(self._alphas)
        # setup beta list
        self._betas = []
        for n, p in self.named_parameters():
            if 'beta' in n:
                self._betas.append((n, p))
        print(self._betas)

    def forward(self, x):
        #weights_1 = F.softmax(self.alpha, dim=-1)
        #weights_2 = F.softmax(self.beta, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, self.alpha,self.beta)
        else:
            raise NotImplementedError
            # multiple GPU support
            # # scatter x
            # xs = nn.parallel.scatter(x, self.device_ids)
            # # broadcast weights
            # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
            # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

            # # replicate modules
            # replicas = nn.parallel.replicate(self.net, self.device_ids)
            # outputs = nn.parallel.parallel_apply(replicas,
            #                                     list(
            #                                         zip(xs, wnormal_copies, wreduce_copies)),
            #                                     devices=self.device_ids)
            # return nn.parallel.gather(outputs, self.device_ids[0])

    def genotype(self):
        return self.net.genotype(self.alpha.cpu().detach().numpy(),self.beta.cpu().detach().numpy())

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def betas(self):
        for n, p in self._betas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def named_betas(self):
        for n, p in self._betas:
            yield n, p

    def print_alphas(self, logger):
        logger.info("####### ALPHA #######")
        for alpha in self.alpha:
            logger.info(F.softmax(alpha, dim=-1).cpu().detach().numpy())
        logger.info("####### BETA #######")
        for beta in self.beta:
            logger.info(F.softmax(beta, dim=-1).cpu().detach().numpy())
        logger.info("#####################")

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)


class Architect():
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get(
                    'momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
            for b, vb in zip(self.net.betas(), self.v_net.betas()):
                vb.copy_(b)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)

        # compute gradient
        v_weights = tuple(self.v_net.weights())
        ####alpha
        v_alphas = tuple(self.v_net.alphas())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights,retain_graph=True)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessiana = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        #with torch.no_grad():

        #####beta
        v_betas = tuple(self.v_net.betas())
        v_bgrads = torch.autograd.grad(loss, v_betas + v_weights)
        dbeta = v_bgrads[:len(v_betas)]
        dbw = v_bgrads[len(v_betas):]

        hessianb = self.compute_hessian(dbw, trn_X, trn_y,1)

        # update final gradient = dbeta - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessiana):
                alpha.grad = da - xi * h
            for beta, db, h in zip(self.net.betas(), dbeta, hessianb):
                beta.grad = db - xi * h



    def compute_hessian(self, dw, trn_X, trn_y,bool_beta = 0):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        if bool_beta == 0:
            dalpha_pos = torch.autograd.grad(
                loss, self.net.alphas())  # dalpha { L_trn(w+) }
        else:
            dalpha_pos = torch.autograd.grad(
                loss, self.net.betas())  # dbeta { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        if bool_beta == 0:
            dalpha_neg = torch.autograd.grad(
                loss, self.net.alphas())  # dalpha { L_trn(w-) }
        else:
            dalpha_neg = torch.autograd.grad(
                loss, self.net.betas())  # dbeta { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class PcMixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, basic_op_list=None):
        super().__init__()

        self.k = 4
        self.mp = nn.MaxPool2d(2, 2)
        self._ops = nn.ModuleList()
        assert basic_op_list is not None, "the basic op list cannot be none!"
        basic_primitives = basic_op_list
        for primitive in basic_primitives:
            op = OPS_[primitive](C_in//self.k, C_out//self.k, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):

        #channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2//self.k, :, :]
        xtemp2 = x[ : ,  dim_2//self.k:, :, :]
        assert len(self._ops) == len(weights)
        #print("#######op(xtemp)")
        #print((self._ops[0](xtemp).size()))
        #print("#######weights")
        #print(weights)
        #temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))

        temp1 = 0
        for i, value in enumerate(weights):
            if value == 1:
                temp1 += self._ops[i](xtemp)
            if 0 < value < 1:
                temp1 += value * self._ops[i](xtemp)

        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
          ans = torch.cat([temp1,xtemp2],dim=1)
        else:
          ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans,self.k)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans


# the search cell in darts


class PcDartsCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, basic_op_list):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.basic_op_list = basic_op_list

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = PcMixedOp(C, C, stride, self.basic_op_list)
                self.dag[i].append(op)
        '''
        self.dag = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self.n_nodes):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = PcMixedOp(C, C, stride, self.basic_op_list)
                self.dag.append(op)'''

    def forward(self, s0, s1, sample,sample2):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        #print("#####sample####")
        #print(type(sample), sample)
        #("#####sample2####")
        #print(type(sample2), sample2)
        states = [s0, s1]
        w_dag = darts_weight_unpack(sample, self.n_nodes)
        w_w_dag = darts_weight_unpack(sample2, self.n_nodes)
        #print("#####w_dag####")
        #print(type(w_dag),w_dag)
        #print("#####w_w_dag####")
        #print(type(w_w_dag),w_w_dag)
        for edges, w_list,w_w_list in zip(self.dag, w_dag,w_w_dag):
            '''print("#####state####")
            print((len(states)))
            print("#####w_list####")
            print(len(w_list),w_list)
            print("#####w_w_list####")
            print(len(w_w_list),w_w_list)'''

            s_cur = sum(ww * edges[i](s, w)
                        for i, (s, w, ww) in enumerate(zip(states, w_list, w_w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], 1)
        return s_out
        '''
        s0 = self.preproc0(s0)
        s1 = self.preproc0(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self.n_nodes):
            s = sum(sample2[offset + j] * self.dag[offset + j](h, sample[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)'''

# PcDartsCNN


class PcDartsCNN(nn.Module):

    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[]):
        super().__init__()
        stem_multiplier = 3
        self._multiplier = 4
        self.C_in = 3  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        self.basic_op_list = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
                              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none'] if len(basic_op_list) == 0 else basic_op_list
        C_cur = stem_multiplier * C  # 3 * 16 = 48
        self.stem = nn.Sequential(
            nn.Conv2d(self.C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C
        # 48   48   16
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = PcDartsCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, self.basic_op_list)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
        # number of edges per cell
        self.num_edges = sum(list(range(2, self.n_nodes + 2)))
        # whole edges
        self.all_edges = 2 * self.num_edges



    def forward(self, x, sample,sample2):
        s0 = s1 = self.stem(x)

        for i,cell in enumerate(self.cells):
            if cell.reduction:
                alphas_reduce = sample[self.num_edges:]
                betas_reduce = sample2[self.num_edges:]
                weights = F.softmax(alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(betas_reduce[0:2], dim=-1)
                for i in range(self.n_nodes - 1):
                    end = start + n
                    tw2 = F.softmax(betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                alphas_normal = sample[0:self.num_edges]
                betas_normal = sample2[0:self.num_edges]
                weights = F.softmax(alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(betas_normal[0:2], dim=-1)
                for i in range(self.n_nodes - 1):
                    end = start + n
                    tw2 = F.softmax(betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            #print("#########################################weights")
            #print(weights)
            #print("#########################################weights2")
            #print(weights2)
            s0, s1 = s1, cell(s0, s1, weights , weights2)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits

    def genotype(self, theta,theta2):

        Genotype = namedtuple(
            'Genotype', 'normal normal_concat reduce reduce_concat')
        a_norm = theta[0:self.num_edges]
        a_reduce = theta[self.num_edges:]
        b_norm = theta2[0:self.num_edges]
        b_reduce = theta2[self.num_edges:]
        weightn = F.softmax(a_norm, dim=-1)
        weightr = F.softmax(a_reduce, dim=-1)
        n = 3
        start = 2
        weightsn2 = F.softmax(b_norm[0:2], dim=-1)
        weightsr2 = F.softmax(b_reduce[0:2], dim=-1)

        for i in range(self.n_nodes - 1):
            end = start + n
            tn2 = F.softmax(b_norm[start:end], dim=-1)
            tw2 = F.softmax(b_reduce[start:end], dim=-1)
            start = end
            n += 1
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)

        theta_norm = darts_weight_unpack(weightn, self.n_nodes)
        theta_reduce = darts_weight_unpack(weightr, self.n_nodes)
        theta2_norm = darts_weight_unpack(weightsn2, self.n_nodes)
        theta2_reduce = darts_weight_unpack(weightsr2, self.n_nodes)

        for t, etheta in enumerate(theta_norm):
            for tt, eetheta in enumerate(etheta):
                theta_norm[t][tt] *= theta2_norm[t][tt]
        for t, etheta in enumerate(theta_reduce):
            for tt, eetheta in enumerate(etheta):
                theta_reduce[t][tt] *= theta2_reduce[t][tt]

        gene_normal = parse_from_numpy(
            theta_norm, k=2, basic_op_list=self.basic_op_list)
        gene_reduce = parse_from_numpy(
            theta_reduce, k=2, basic_op_list=self.basic_op_list)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes
        return Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)



