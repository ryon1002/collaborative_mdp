# MDPメモ

## Base POMDP

### 簡単定義

$<\mathcal{S}, \mathcal{A}, \mathcal{Z}, \mathcal{T}, \mathcal{O}, \mathcal{R}>$

$T = p(s'|s, a)$

$O = P(z |s, a)$ $a$をとって$s$に到着したときの$z$の観測確率

### アルゴリズムメモ

$\alpha^{a, z}(s)=\sum_{s'}P(s'|s,a)P(z |s', a)\alpha(s')$ 状態遷移を加味したアルファベクターの遷移(次の状態のアルファベクター * 状態遷移確率 * 観測確率)

※結局の所、本質は $\alpha(s)=r(s) + \gamma\sum_s\sum_zP(s', z| s, a)\alpha(s')$。行動$(s, a)$に対して観測、次状態$(s', z)$の確率

$\Gamma^{a, z}=r^a/|\mathcal{Z}|+\gamma \alpha^{a, z}$ 報酬の追加(あとで観測で足すので観測数で割ってる)

$\Gamma^{a}=\bigoplus_z\Gamma^{a, z}$ 全観測の組み合わせにおいて和集合をアルファベクターの候補にする

$\Gamma = prune(\bigcup_a \Gamma^a)$ 全行動の候補を集合合算させて枝刈り



### その他

$$
b^{a, z}(s')=p(s'|b, a, z)=\frac{P(s', b, a, z)}{P(b, a, z)}=\frac{P(z|s', a)P(s'|b, a)}{P(z|b, a)}=\frac{P(z|s', a)\sum_s P(s'|s, a)b(s)}{P(z|b, a)}
$$

## Base MOMDP

### 簡単定義

$<\mathcal{X}, \mathcal{Y}, \mathcal{A}, \mathcal{Z}, \mathcal{T}, \mathcal{O}, \mathcal{R}>$

$T = p(s'|s, a)$

$O = P(z |s, a)$ $a$をとって$s$に到着したときの$z$の観測確率

### アルゴリズムメモ

$\alpha^{a, z}(s)=\sum_{s'}P(s'|s,a)P(z |s', a)\alpha(s')$ 状態遷移を加味したアルファベクターの遷移(次の状態のアルファベクター * 状態遷移確率 * 観測確率)

$\Gamma^{a, z}=r^a/|\mathcal{Z}|+\gamma \alpha^{a, z}$ 報酬の追加(あとで観測で足すので観測数で割ってる)

$\Gamma^{a}=\bigoplus_z\Gamma^{a, z}$ 全観測の組み合わせにおいて和集合をアルファベクターの候補にする

$\Gamma = prune(\bigcup_a \Gamma^a)$ 全行動の候補を集合合算させて枝刈り

---
$\alpha^k_{s,a,z}=\sum_{s'} \alpha^k_{s'} p(z|s')p(s'|s,a)$

$Z = p(z | s)$：$s$にいるときの$z$の観測確率

#$V_T(b)=max_u{V_T(b,u)}=\gamma max_u([\sum_{i=1}^Np_i r(s_i ,u)]+\sum_z max_k \sum_{i=1}^Np_i \sum_{j=)$

