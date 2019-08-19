import torch
from root_finding import TsallisBisectAlphaFunction
from torch.autograd import grad
# from torch.autograd.gradcheck import get_numerical_jacobian
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3)


def check_backward():
    torch.manual_seed(42)
    x = torch.randn((2, 6), requires_grad=True)
    alpha = 1 + torch.rand((1, ), requires_grad=True).squeeze()

    # print(x)
    print(alpha)

    f = TsallisBisectAlphaFunction.apply
    p = f(x, alpha)
    # print(p)

    def fa(alpha):
        return f(x, alpha)

    # print(get_numerical_jacobian(fa, alpha))
    eps = 1e-5
    p_p = fa(alpha + eps)
    p_m = fa(alpha - eps)

    da_numeric = (p_p - p_m) / (2 * eps)
    # print(da_numeric)

    dP = torch.zeros_like(p)
    flat_dP = dP.view(-1)

    da = torch.zeros_like(p)
    flat_da = da.view(-1)

    for i in range(flat_dP.numel()):
        flat_dP.zero_()
        flat_dP[i] = 1
        ret = grad(p, alpha, dP, retain_graph=True)
        # print(dP, ret)
        flat_da[i] = ret[0]

    # print(da)

    print(((da - da_numeric) ** 2).sum())


def main():
    torch.manual_seed(42)
    x = torch.randn((1, 6), requires_grad=True)

    f = TsallisBisectAlphaFunction.apply

    alphas = torch.tensor([1.3, 1.7], requires_grad=True)
    a_1 = alphas[0]
    a_2 = alphas[1]
    p_1 = f(x, a_1)
    p_2 = f(x, a_2)
    p_expected = torch.cat([p_1, p_2], dim=0)
    print(p_expected)

    x_repeated = torch.cat([x, x], dim=0)

    p = f(x_repeated, alphas)
    # print(p)
    dP_1 = torch.randn_like(p_1)
    dP_2 = torch.randn_like(p_2)

    ret_1 = grad(
        p_1, a_1, dP_1, retain_graph=True,
        allow_unused=True)
    ret_2 = grad(
        p_2, a_2, dP_2, retain_graph=True,
        allow_unused=True)
    print(ret_1)
    print(ret_2)

    dP = torch.cat([dP_1, dP_2], dim=0)
    ret = grad(
        p, alphas, dP, retain_graph=True,
        allow_unused=True)
    print(ret)


def real_world():
    torch.manual_seed(42)
    batch_size = 16
    head_count = 4
    query_len = 12
    key_len = 10

    f = TsallisBisectAlphaFunction.apply

    x = torch.randn(
        (batch_size,
         head_count,
         query_len,
         key_len), requires_grad=True)

    alpha = 1 + torch.rand((head_count, ), requires_grad=True).squeeze()
    print(alpha)

    out = torch.zeros(
        (batch_size,
         head_count,
         query_len,
         key_len))

    if True:
        for i in range(batch_size):
            for j in range(head_count):
                for k in range(query_len):
                    out[i, j, k] = f(x[i, j, k].unsqueeze(0), alpha[j])

        dout = torch.randn_like(out)
        g = grad(
            out, alpha, dout, retain_graph=True,
            allow_unused=True)
        print(g)

    def broadcaster(x, alpha):
        expanded_alpha = alpha.unsqueeze(0).unsqueeze(-1)
        expanded_alpha = expanded_alpha.expand((batch_size, -1, query_len))

        # x_cont = x.contiguous().
        x_cont = x.view(-1, key_len)
        expanded_alpha = expanded_alpha.contiguous().view(-1)

        normalized = f(x_cont, expanded_alpha)
        return normalized.view_as(x)

    out2 = broadcaster(x, alpha)
    print(torch.norm(out2 - out))
    g = grad(
        out2, alpha, dout, retain_graph=True,
        allow_unused=True)
    print(g)


if __name__ == "__main__":
    real_world()
