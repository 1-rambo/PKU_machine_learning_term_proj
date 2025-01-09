#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

torch::Tensor URAN(int n);
torch::Tensor GRAN(int n, int m);
torch::Tensor ORTH(torch::Tensor B);
torch::Tensor RED(torch::Tensor B);
torch::Tensor CLP(int n, torch::Tensor B, torch::Tensor x);
Eigen::MatrixXf LLL(Eigen::MatrixXf v);

// int main() {
//     // for test
//     torch::Tensor B = torch::randn({5, 5});
//     torch::Tensor x = torch::randn({5});
//     std::cout << "*1* Generated URAN (5): " << URAN(5) << std::endl;
//     std::cout << "*2* Generated GRAN (3, 4): " << GRAN(3, 4) << std::endl;
//     std::cout << "*3* Orthogonal matrix from B: " << ORTH(B) << std::endl;
//     std::cout << "*4* Closest lattice point: " << CLP(5, B, x) << std::endl;
//     return 0;
// }

torch::Tensor URAN(int n){
    return torch::rand(n);
}

torch::Tensor GRAN(int n, int m) {
    return torch::randn({n, m});
}

torch::Tensor ORTH(torch::Tensor B) {
    torch::Tensor A = torch::matmul(B, B.transpose(0, 1));
    return torch::linalg::cholesky(A);
}

torch::Tensor RED(torch::Tensor B) {
    auto B_np = B.detach().cpu().to(torch::kFloat32);
    Eigen::Map<Eigen::MatrixXf> B_eigen(B_np.data_ptr<float>(), B.size(0), B.size(1));
    Eigen::MatrixXf reduced = LLL(B_eigen);
    
    auto result = torch::from_blob(reduced.data(), {reduced.rows(), reduced.cols()}, torch::kFloat32).clone();
    return result;
}

int sign(float x) {
    return (x > 0) ? 1 : -1;
}

torch::Tensor CLP(int n, torch::Tensor B, torch::Tensor x) {
    float C = std::numeric_limits<float>::infinity();
    int i = n;
    torch::Tensor d = torch::full({n}, n - 1);
    torch::Tensor lamb = torch::zeros({n + 1});
    torch::Tensor u = torch::zeros({n});
    torch::Tensor p = torch::zeros({n});
    torch::Tensor Delta = torch::zeros({n});
    torch::Tensor result = torch::zeros({n});
    torch::Tensor F = torch::zeros({n, n});
    F[n - 1] = x.clone();
    // F.index_put_({n - 1}, x.clone());


    while (true) {
        while (true) {
            if (i != 0) {
                i = i - 1;
                for (int j = d[i].item<int>(); j > i; j--) {
                    // F[j - 1][i] = F[j][i] - u[j] * B[j][i];
                    float val = F.index({j, i}).item<float>() - u[j].item<float>() * B.index({j, i}).item<float>();
                    F.index_put_({j - 1, i}, val);
                }
                // p[i] = F[i][i] / B[i][i];
                p.index_put_({i}, F.index({i, i}) / B.index({i, i}));
                // u[i] = torch::round(p[i]);
                u.index_put_({i}, torch::round(p[i]));
                float y = (p[i].item<float>() - u[i].item<float>()) * B.index({i, i}).item<float>();
                Delta.index_put_({i}, 1.0 * sign(y));
                lamb.index_put_({i}, lamb.index({i + 1}).item<float>() + y * y);
            }
            else {
                result = u.clone();
                C = lamb[0].item<float>();
            }
            if (lamb[i].item<float>() >= C) {
                break;
            }
        }
        int m = i;
        while (true) {
            if (i == n - 1) {
                return result;
            } else {
                i = i + 1;
                // u[i] = u[i] + Delta[i];
                u.index_put_({i}, u.index({i}).item<float>() + Delta.index({i}).item<float>());
                // Delta[i] = -Delta[i] - 1;
                Delta.index_put_({i}, -Delta.index({i}).item<float>() - 1);
                float y = (p[i].item<float>() - u[i].item<float>()) * B[i][i].item<float>();
                // lamb[i] = lamb[i + 1] + y * y;
                lamb.index_put_({i}, lamb.index({i + 1}).item<float>() + y * y);
            }
            if (lamb[i].item<float>() < C) {
                break;
            }
        }
        for (int j = m; j < i; j++) {
            // d[j] = i;
            d.index_put_({j}, i);
        }
        for (int j = m - 1; j >= 0; j--) {
            if (d[j].item<int>() < i) {
                d[j] = i;
            } else {
                break;
            }
        }
    }
}

Eigen::MatrixXf orthogonal(const Eigen::MatrixXf& m) {
    int n = m.rows();
    int d = m.cols();
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(n, d);
    M.row(0) = m.row(0);

    for (int i = 1; i < n; ++i) {
        M.row(i) = m.row(i);
        for (int j = 0; j < i; ++j) {
            double u_ij = (m.row(i).dot(M.row(j))) / (M.row(j).squaredNorm());
            M.row(i) -= u_ij * M.row(j);
        }
    }
    return M;
}

Eigen::MatrixXf lll(Eigen::MatrixXf v) {
    int n = v.rows();
    int k = 2;
    
    while (k <= n) {
        Eigen::MatrixXf V = orthogonal(v.topRows(k));
        
        for (int j = 0; j < k - 1; ++j) {
            double u = (v.row(k - 1).dot(V.row(j))) / (V.row(j).squaredNorm());
            v.row(k - 1) -= round(u) * v.row(j);
        }

        double u = (v.row(k - 1).dot(V.row(k - 2))) / (V.row(k - 2).squaredNorm());
        if (V.row(k - 1).squaredNorm() >= (3.0 / 4.0 - u * u) * V.row(k - 2).squaredNorm()) {
            k++;
        } else {
            v.row(k - 2).swap(v.row(k - 1));
            k = std::max(k - 1, 2);
        }
    }
    return v;
}

Eigen::MatrixXf LLL(Eigen::MatrixXf v) {
    Eigen::MatrixXf a = lll(v);
    Eigen::MatrixXf b = lll(a);
    // printf("LLL\n");

    while (!a.isApprox(b)) {
        a = b;
        b = lll(b);
    }

    return b;
}