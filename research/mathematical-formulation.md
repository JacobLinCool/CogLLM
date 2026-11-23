# CogLLM 的數學規格

CogLLM 的目標是以既有資料重現指定讀者群的可觀測反應。對一篇文章，Qwen3.5-2B 一次因果前向運算，讀出頭同時估計各 token 的反應機率、反應所指區段，以及有對應量測時的強度。分析程式比較同一材料的不同版本，並以整組留出的既有觀測驗證可比端點。本文記錄逐 token 的設計規格及第 31 輪聯合讀出的實作範圍。 [研究方案](training-protocol.md) · [第 31 輪的定位結果](union_training/localization-results.md)

## 1. 輸入、讀者與座標

令 $X$ 為全文，$x_{1:T}=\operatorname{Tok}(X)$ 為正文 token 序列，$I_t=[b_t,e_t)$ 為 token $t$ 對應的原文字元區間。$c$ 為可選的讀者背景，只使用閱讀前可取得的資訊。令 $C(c)$ 為含固定起始標記與分隔符的前綴，$P=|C(c)|$，$j_t=P+t$。整次輸入為

$$
z_{1:N}=C(c)\mathbin{\Vert}x_{1:T},\qquad N=P+T,\qquad j_0=P.
$$

背景與正文分開編碼再串接，以維持正文偏移的對齊。$C(\varnothing)$ 只含固定標記，因此第一個正文 token 也有前一位置。按原研究實際呈現的文字記錄已觀測停點。若停點切過 token，保留區間或另以實際可見的字串編碼該試次，避免用包含未呈現字元的表示作為該停點的輸入。

對每種量測 $m$ 固定報告指示、可見文本和回看規則。它們定義輸出頭對應的觀察條件。一般介面以正文和選填背景為輸入，研究題目、版本標記、答案和真人回報保留在研究端。$c=\varnothing$ 的反應機率以指定參考讀者群 $P_{\mathrm{ref}}$ 校準。以事件 $Y$ 為例，其目標為

$$
p^*_{m,t,k}(X,\varnothing)
=\mathbb E_{C\sim P_{\mathrm{ref}}}
  \big[\Pr_m(Y_{t,k}=1\mid x_{\le t},C)\big].
$$

規格實作可直接用該群的既有樣本訓練空背景頭。同一試次若已知背景且屬參考群，訓練時可以遮去 $c$ 來監督空背景輸出。若有背景的受試者分布與參考群不同，須以既有的可比樣本或已知權重校準。空背景表示缺少個人資料。

## 2. 一次因果前向運算

以 $h^{(\ell)}_i\in\mathbb R^D$ 表示第 $\ell$ 個 block 輸出、末端 RMSNorm 前的表徵，$D=2048$。以 $A$ 表示擷取的 full-attention 權重，$G$ 表示選用的 Gated DeltaNet 訊號。一次執行

$$
\big(H^{(1)},\ldots,H^{(24)},A,G\big)
=F_\theta(z_{1:N}),\qquad
H^{(\ell)}\in\mathbb R^{N\times D}.
$$

讀出頭實際只留所選層和約定的統計量。$h^{(\ell)}_{j_t}$ 依賴 $z_{\le j_t}$。Qwen 的注意力遮罩阻止位置 $i$ 取用 $j>i$ 的 key，Gated DeltaNet 的卷積和狀態更新也只用當下與更早的輸入。逐位置的正規化和前饋不引入其他位置。逐層套用此關係，所有 block 的輸出皆具前綴依賴。正文只走一次模型，先前的反應標籤無需回填，也沒有逐步產生文字的解碼迴圈。 [Qwen3.5-2B 架構](https://huggingface.co/Qwen/Qwen3.5-2B#model-overview) · [Qwen 實作](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

第 31 輪聯合讀出的骨幹前向計算全部 2,598,224 個輸入 token，只保存 164,220 個觀測停點的末層向量。Frick 使用各呈現句的句末；PLABA 的事後評句使用目標句末與文章末；CompLex 和圈選資料亦有指定詞末。前向呼叫的 `output_attentions` 與 `output_hidden_states` 均為 `False`。原資料的停點決定該研究能對齊的真人反應。逐 token 讀出的特徵與輸出網格應獨立於各來源的標註網格。 [前向與擷取](backbone_transfer/extract.py#L295) · [聯合執行收據](results/union-observations-settled-l4/run-verification.json) · [現有定位結果](union_training/localization-results.md)

### Full attention

Qwen3.5-2B 的 full-attention block 為 $\mathcal L_A=\{4,8,12,16,20,24\}$，每層有 8 個 query heads、2 個 key/value heads，維度 $d_h=256$。令 $\widetilde q^{\ell,a}_i,\widetilde k^{\ell,b}_j$ 為經 head RMSNorm 與 RoPE 的 query 和 key。令 $\rho(a)=\lfloor(a-1)/4\rfloor+1$ 將 query head 映射到其 KV head。對單篇無 padding 的輸入，

$$
d_{ij}^{\ell,a}=\frac{(\widetilde q_i^{\ell,a})^\top\widetilde k_j^{\ell,\rho(a)}}{\sqrt{d_h}},
\qquad
A_{ij}^{\ell,a}
=
\begin{cases}
\exp\!\left(d_{ij}^{\ell,a}-
\operatorname{LSE}_{u\le i}d_{iu}^{\ell,a}\right), & j\le i,\\
0, & j>i.
\end{cases}
$$

$d_{ij}^{\ell,a}$ 是 softmax 前的 attention score。Qwen 的輸出 gate 施加在加權 value 上。以下用經 softmax 的權重構成特徵。以 $r=[s,e]$ 表示正文的連續 token 區段。令 $\alpha^{\ell,a}_{t,r}$ 為位置 $t$ 對已讀區段 $r$ 的 attention 質量，$E^{\ell,a}_t$ 為該 query 的 attention entropy。

$$
\alpha^{\ell,a}_{t,r}=\sum_{k=s}^{e} A_{j_t,j_k}^{\ell,a},
\qquad E^{\ell,a}_t=-\sum_{u\le j_t} A_{j_t,u}^{\ell,a}\log A_{j_t,u}^{\ell,a},
\qquad e\le t .
$$

分母涵蓋完整可見前綴，含背景與固定標記。可加入對最近 $w$ 個正文 token 的質量或其他事前定義的摘要；位置與區段都由當下已讀文字決定。若執行後端的融合算子只回傳加權 value，需從同一次運算的 Q/K 擷取或重建所選權重。 [Full-attention 計算](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L646-L736)

### Gated DeltaNet

其餘 18 層可直接取 hidden states，或用同次 forward 的 gate。省略層和 head 索引，對 key 向量 $k_i$、value 向量 $v_i$、保留因子 $\gamma_i$ 和寫入係數 $\beta_i$，每個 head 的核心狀態更新為

$$
S_i=\gamma_i S_{i-1}(I-\beta_i k_i k_i^\top)+\beta_i v_i k_i^\top .
$$

例如可取 $\gamma_i$、$\beta_i$ 或 $\|S_i-S_{i-1}\|_F$ 的 head 摘要。這些是模型內部狀態的特徵。對人類疑問何時暫擱或回復的解釋須由獨立量測支持。 [Gated Delta Rule，式 10](https://arxiv.org/html/2412.06464v3#S3.SS1)

### 詞彙 logits 與中間層投影

令 $U\in\mathbb R^{|\mathcal V|\times D}$ 為詞彙讀出矩陣，$\mathcal N$ 為 Qwen 最後的 RMSNorm。對選定中間層與最後一層取

$$
\ell_i^{(\ell)}=U\mathcal N(h_i^{(\ell)}),\qquad
\pi_i^{(\ell)}=\operatorname{softmax}_{\mathcal V}(\ell_i^{(\ell)}).
$$

$\ell^{(24)}$ 是語言模型的下一 token logits。中間層的 $\ell^{(\ell)}$ 是額外的 *logit lens* 讀出。若後端已回傳最後正規化的表徵，直接套 $U$。可另用在開發資料估計的 affine translator，形成 *tuned lens*。詞彙有 248,320 個索引。計算精確機率時，log-sum-exp 包含全部詞彙。 [Qwen 官方詞彙頭](https://huggingface.co/Qwen/Qwen3.5-2B#model-overview) · [Logit lens 與 tuned lens，§2–3](https://arxiv.org/html/2303.08112#S2)

讀入 $x_t$ 前，語言模型對這個 token 的 surprisal 來自 $j_t-1$。讀完 $x_t$ 後，可使用位置 $j_t$ 對未來詞彙分布的熵，以及同一位置不同層的分布差。

$$
\begin{aligned}
\operatorname{Surp}^{(\ell)}_t
  &=\operatorname{LSE}_{v\in\mathcal V}\ell_{j_t-1,v}^{(\ell)}
    -\ell_{j_t-1,x_t}^{(\ell)},\\
\operatorname{Ent}^{(\ell)}_t&=-\sum_v\pi_{j_t,v}^{(\ell)}\log\pi_{j_t,v}^{(\ell)},\\
\operatorname{Div}^{(\ell)}_t
  &=D_{\mathrm{KL}}\!\left(\pi^{(24)}_{j_t}\,\|\,\pi^{(\ell)}_{j_t}\right).
\end{aligned}
$$

前向後已知全文，仍只用 $x_{\le t}$ 提取位置 $t$ 的特徵。$\ell_{j_t,x_{t+1}}$ 需知道尚未讀到的真實 token，因此不能作為 $t$ 的輸入。實作可分塊計算並立即縮減注意力與詞彙統計量。完整存下六層 attention 需要 $48N^2$ 個值，詞彙 logits 每層需要 $N|\mathcal V|$ 個值。

## 3. 平行讀出反應、位置與強度

令 $\phi_t=\Phi(h_{j_t}^{(\ell)},E_t,G_t,\operatorname{Surp}_t,\operatorname{Ent}_t,\operatorname{Div}_t)$ 為位置 $t$ 的特徵向量。$\Phi$ 只使用在讀到 $t$ 時可得的特徵，含選用的 attention 統計量。初版以各層 hidden states、attention 和詞彙特徵的事先選定子集為候選。讀出頭權重的估計、特徵標準化和門檻都使用開發資料。

### 每個 query 對已讀位置的關係

對正文各 token 取得其定位座標、末層及選定中間層向量，保留六個 full-attention block 的逐頭關係 $\{A^{\ell,a}_{j_t,i}\mid i\le j_t\}$。讀出可就任一已讀 token 或區段取用相應 attention 權重及該處內容表徵。以分塊計算或從同一次前向保留的 Q/K 重建，正規化分母仍取完整可見前綴。對 18 個 Gated DeltaNet block，逐位置保留所選層輸出、gate 或狀態更新訊號。這個方案保留每個 query 所指的實際位置，因此可檢驗模型能否區分相同目標詞在不同前文下的反應。 [固定 Qwen 設定](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json#L269) · [骨幹與前綴計算](#2-一次因果前向運算)

既有 Frick 實驗在每個 token 擷取末一個 full-attention block 的訊號，跨 head 彙整，再彙整到句級。加入這類摘要後，首次回報整篇 NLL 有改善，目標句的條件差幅改善區間仍跨零。PLABA 的六層摘要也以句末 query 匯集。保留逐頭逐位置關係的讀出是本文待驗收的設計。 [Frick 注意力摘要實驗](semantic_tracking/frick-signals-results.md) · [PLABA 注意力摘要實驗](stage-3-feature-ablation.md)

驗收以同一資料和因果前綴比較純逐 token 向量的讀出及加入原生 attention 關係的讀出。兩者使用相同的訓練集回報基準率及各讀出的獨立收斂檢查，以分辨前文訊號的增益與機率校準的改善。Frick 的似然於原本的句末回報處計算，可聚合整段句內狀態。逐詞觸發點、所指上文及解除時間依各自可對位的真人結果或實驗操弄驗收。所有位置先通過相同前綴條件的截短、後綴替換及 padding 檢查。 [首次回報讀出診斷](union_training/localization-results.md#frick-讀出在整體停訓時的擬合程度)

### 反應與所指區段

先依真人的回報規則定義 $K$ 類反應。例子包括仍有指涉疑問、感到跨句接不上、已能確認前文指涉。反應類別可以共存。記 $Y_{t,k}=1$ 為讀者在停點 $t$ 依該規則回報類型 $k$，$Z_{t,r,k}=1$ 為該回報明確指向已讀區段 $r$。$Z_{t,r,k}=1$ 蘊含 $Y_{t,k}=1$。$t$ 和 $r$ 分別記錄反應的觀測位置、反應所指的文字。實驗可另標操弄的觸發線索；它與真人指出的區段可不同。

給定最大候選長度 $B$，可用 $\mathcal R_t=\{[s,e]:1\le s\le e\le t,\ e-s+1\le B\}$。若另加候選篩選，其可見輸入受同一前綴限制。以 $u_r=[h^{(24)}_{j_s};h^{(24)}_{j_e};(e-s+1)^{-1}\sum_{v=s}^{e}h^{(24)}_{j_v}]$ 為區段表示。$u_r$ 可用端點與前綴和一次計算。以下是一個可直接實作的讀出，其中 $\alpha_{t,r}$ 串接選定 layers 和 heads 的質量，$d(t,r)$ 為已讀距離特徵，$D_p$ 為雙線性投影維度。

$$
\begin{aligned}
p_{t,k}&=\sigma(w_k^\top\phi_t+b_k),\\
q_{t,r,k}
&=\sigma\!\left(
   \frac{(W_{Q,k}\phi_t)^\top(W_{R,k}u_r)}{\sqrt{D_p}}
   +a_k^\top\alpha_{t,r}+v_k^\top d(t,r)+b'_k
   \right),\qquad r\in\mathcal R_t .
\end{aligned}
$$

$p_{t,k}$ 估計 $\Pr(Y_{t,k}=1)$，$q_{t,r,k}$ 估計 $\Pr(Z_{t,r,k}=1\mid Y_{t,k}=1)$。同時回報 $k$ 且指向 $r$ 的機率為 $J_{t,r,k}=p_{t,k}q_{t,r,k}$。多個區段可同時成立，各 $q$ 不需要總和為 1。若只在能定位的報告中取得具代表性的區段標註，只能估計該子群的 $q$。此時跨全部回報的 $J$ 須有額外的定位率或抽樣資訊才可估計。

〔構造案例〕假設讀到下一句的第一個關鍵詞時，回報「前文接不上」的機率是 $p=0.60$。在有這類回報的人中，指向上一句「那張卡」的比例為 $q=0.75$。則兩者共同發生的機率為 $J=0.45$。觀測位置在下一句，所指區段在上一句。

所有 $p$ 和合法配對的 $q$ 均可用張量運算平行評分。事件輸出大小為 $T\times K$。區段評分採稀疏索引 $(t,r,k)$，可分塊估計。若枚舉所有長度至多為 $B$ 的區段及其合法位置，配對數為 $O(BT^2)$。候選受限時，需另外報真人金標區段的涵蓋率。

### 獨立的強度量尺

$p_{t,k}$ 的單位是回報機率。若真人另有 $L$ 級的心智努力或不確定感評分 $R^{(m)}_t$，用獨立的序位頭估計該量表。令 $\eta^{(m)}_t=v_m^\top\phi_t$，用由低至高的切點 $\kappa_1<\cdots<\kappa_{L-1}$ 定義

$$
F_{t,m}(\ell)=\Pr(R^{(m)}_t\le\ell)=\sigma(\kappa_\ell-\eta^{(m)}_t),\qquad
F_{t,m}(0)=0,\quad F_{t,m}(L)=1.
$$

第 $\ell$ 級機率是 $F_{t,m}(\ell)-F_{t,m}(\ell-1)$，平均分數為 $\sum_{\ell=1}^L\ell[F_{t,m}(\ell)-F_{t,m}(\ell-1)]$。若量尺只問有疑問的讀者，讀出也以有疑問為條件。句末自評以句末表徵監督，全文後的回顧量測用全文可見時的表徵加專用頭。個別理解作答與閱讀時間可作輔助端點，各自保留觀測單位與損失。

### 句級輸出與 mask

令 $S_j$ 為句子 $j$ 的原文字元區間。句內峰值摘要可用 $\max_{t:I_t\cap S_j\ne\varnothing}p_{t,k}$，單位仍為單一位置的分數。若研究要估計讀者實際回報「這一句有困難」的比例，另用句級真人標註校準該端點。需顯示高亮時，用開發資料選定門檻 $\xi,\zeta$。令 $I$ 為待渲染的原文區間，例如單一 token 或整句，$I(r)$ 為候選區段的字元範圍。

$$
m^{\mathrm{event}}_{t,k}=\mathbf 1\{p_{t,k}\ge\xi_k\},\qquad
m^{\mathrm{source}}_{t,k}(I)
=\mathbf 1\{\exists r\in\mathcal R_t:J_{t,r,k}\ge\zeta_k,\ I\cap I(r)\ne\varnothing\}.
$$

前者標記何時有反應，後者標記反應指向的原文。若區段頭只對可定位子群校準，顯示的區段分數須保留該條件。訓練中另有監督遮罩，記錄哪個資料欄位實際有觀測。

## 4. 人類觀測與聯合訓練

令 $i$ 為真人試次，對應材料 $X_i$、背景 $c_i$、可見停點及採用的回報規則。一篇文章可以有多位讀者和多個停點。相同模型輸入可共用特徵，保留各筆真人反應以計入分母及抽樣變異。自由回報依事先定義的規則編成類別和所指區段。可用的公開起點是 INQUISITIVE 的句末問題及自選區段。其標註含讀者想問的問題，可用來學已取得的疑問與位置。要校準一般讀者的回報率、確認在哪一詞開始卡住或何時解除，需有相應的抽樣、負例與追蹤。 [INQUISITIVE，§3.2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3)

### 事件和區段的觀測似然

單一停點和反應類型可用下式建模。令 $\mathbf z$ 為所有合法區段的二元標註向量，$\operatorname{Ber}(y;p)=p^y(1-p)^{1-y}$。

$$
\Pr(Y=y,\mathbf Z=\mathbf z\mid x_{\le t},c)=
\begin{cases}
1-p & y=0,\ \mathbf z=\mathbf 0,\\
0 & y=0,\ \mathbf z\ne\mathbf 0,\\
p\displaystyle\prod_{r\in\mathcal R_t}\operatorname{Ber}(z_r;q_r) & y=1 .
\end{cases}
$$

乘積是初版用於同一次回報內多區段的工作模型。若只使用各區段的 BCE，可將它解讀為邊際複合似然。主要比較取單一區段的 $q$ 或 $J$，避免依賴全體區段的條件獨立假設。多類反應的邊際可共用表徵和聯合訓練。若研究要比較兩類是否共同出現，需該共同端點的資料與讀出。

令 $\Omega_Y$ 為反應類型的正負已知、且抽樣可用於估計反應率的索引 $(i,t,k)$。令 $\Omega_Z$ 為 $Y=1$ 且區段是否被指向已知的索引 $(i,t,r,k)$。令 $\Omega_m$ 為第 $m$ 個量測的有效觀測索引。以 $w_o>0$ 表示事先指定的抽樣或研究權重，$n(\Omega)=\sum_{o\in\Omega}w_o$，則

$$
\begin{aligned}
\mathcal L_Y
 &=\frac{1}{n(\Omega_Y)}\sum_{(i,t,k)\in\Omega_Y}
   w_{i,t,k}\operatorname{BCE}(y_{i,t,k},p_{i,t,k}),\\
\mathcal L_Z
 &=\frac{1}{n(\Omega_Z)}\sum_{(i,t,r,k)\in\Omega_Z}
   w_{i,t,r,k}\operatorname{BCE}(z_{i,t,r,k},q_{i,t,r,k}),\\
\mathcal L_m
 &=\frac{1}{n(\Omega_m)}\sum_{o\in\Omega_m}w_o[-\log P_{\psi,m}(y_o\mid \Phi_o)].
\end{aligned}
$$

$\operatorname{BCE}(y,p)=-y\log p-(1-y)\log(1-p)$。$\Phi_o$ 依量測時實際可見的內容及評估對象取值。逐停點回報取 $\phi_t$。對已讀句子 $r$ 的即時評分取 $[\phi_{e(r)};u_r]$。讀完整篇後回顧句子 $r$ 的評分取 $[\phi_T;u_r]$。篇章端點可取 $\phi_T$。這些向量都取自同一次 forward。序位強度用上一節的級別機率，二元端點用 Bernoulli。若只有同條件的 $a$ 筆有反應與 $n-a$ 筆無反應，該端點可用 $-a\log p-(n-a)\log(1-p)$，並保留原研究的抽樣單位作推論。句級或篇章端點各自監督句級或篇章輸出。

- 明確完成回報且說沒有疑問，可記為該回報規則下的負例。停點未取樣、未完成回報或區段無法判定時，從相應的損失排除。
- 只有要求產生疑問的正例資料可用於學習已回報疑問的類型及所指區段。若取樣只保留 $Y=1$ 的例子，缺乏母體分母，該筆資料不進 $\Omega_Y$。只標出一部分區段時，其他區段也需有明確標註才計為負例。
- 區段條件機率的母體依抽樣規則定義。假如 $L=1$ 表示報告明確可定位，且未能定位的報告依定義沒有可指認的區段，可以另估 $\Pr(L=1\mid Y=1)$，把它乘上 $p$ 和 $\Pr(Z_r=1\mid Y=1,L=1)$。若只是缺少標註，依既有資料的觀測機制判斷能否推廣到全體。可對齊的原始回報由 agent 重編碼，仍無足夠資料時將估計限在可觀測群體。

在觀測有支撐且抽樣適用的條件下，Bernoulli 的期望 BCE 等於真實分布的熵加上它到模型分布的 KL divergence。因此期望損失在真實回報機率取得最小。凍結表徵是否保留估計該機率所需的訊息，需由留出人類資料檢查。句末資料只提供句末的約束。未採樣的 token 上，輸出可以提出更細的定位假說。

### 從文獻效應回傳梯度

把一筆文獻效應寫成原研究在指定材料、讀者群、可見內容和測量規則下的對比。令 $\mathcal I_{j,a}$ 為研究 $j$ 的條件 $a$ 下可評估的模型輸入和讀出位置，$w_{j,a,i}$ 為依研究估計方法設定的歸一化權重。令 $f_j(Y)$ 為原研究量到的值，例如續寫是否指向第一個人物，或主觀努力量表上選了第幾級。模型的條件均值與效應為

$$
\widehat\mu^M_{j,a}=
\sum_{i\in\mathcal I_{j,a}}w_{j,a,i}
\sum_y f_j(y)\,P_{\theta,\psi,m(j)}(Y_i=y\mid X_i,c_i),\qquad
\boldsymbol\delta^M_j=C_j\widehat{\boldsymbol\mu}^M_j .
$$

$m(j)$ 選取與研究端點相符的讀出。若 $Y$ 為連續變數，內層和式改用期望積分。$C_j$ 的一列可以是兩條件的差，或先備知識 × 文章連貫性的差中之差。若原文使用按讀者平均、指定各文章等權、轉換量尺或混合模型係數，記下原分析並先核對資料能否重算。上式適用於以條件均值定義的對比。其他統計量須事先定義與原文一致的 $g_j$。複雜端點可用逐次似然訓練，並保留原統計量供評估。

對有原刺激與適用估計精度的開發研究，令 $\widehat{\boldsymbol\delta}^{\,H}_j$ 為真人的估計，$\widehat\Sigma_j$ 為其估計共變異矩陣。可用

$$
\mathcal L_{\mathrm{effects}}=\sum_{j\in\mathcal J_{\mathrm{dev}}}
  \big[\boldsymbol\delta^M_j-\widehat{\boldsymbol\delta}^{\,H}_j\big]^\top
  \widehat\Sigma_j^{-1}
  \big[\boldsymbol\delta^M_j-\widehat{\boldsymbol\delta}^{\,H}_j\big].
$$

單一對比的權重是估計變異的倒數。重複量測或共享試次的效應需要共變異，否則首輪選定一項主要對比訓練，其餘供診斷。若同一試次已用於逐次損失，初版將從這些試次彙整的效應留作開發診斷。只有方向時，在原比較條件可重建的情況下可做事先指定的方向診斷。未顯著的結果保留其精度。缺原材料或相應端點時標為不可檢驗，估計精度不足時保留其不確定性。

$\boldsymbol\delta^M_j$ 用模型的機率或序位分布直接算，因而可對讀出頭及適用時的 LoRA 回傳梯度。訓練一個效應需前向執行其各條件的文本，每個文本／讀者條件仍是單次 Qwen forward。凍結 Qwen 時，先快取各條件的表徵，在完整對比上更新小型讀出頭。文獻效應保留其統計粒度。就算兩種模型重現相同的條件率差，它們各條件的絕對機率仍可能不同，也可能在不同 token 分配難度。逐次標註與留出定位專門查驗這兩項。以弱標註約束模型輸出的期望已有方法先例。 [Mann & McCallum，§2](https://jmlr.org/papers/v11/mann10a.html) · [Liang et al.，§2](https://www-cs.stanford.edu/people/pliang/papers/measurements-icml2009.pdf)

### 聯合目標

初版凍結 $\theta=\theta_0$，以讀出參數 $\psi$ 最小化

$$
\widehat\psi=\arg\min_\psi
\left[
\lambda_Y\mathcal L_Y+\lambda_Z\mathcal L_Z+
\sum_m\lambda_m\mathcal L_m+
\lambda_E\mathcal L_{\mathrm{effects}}+
\lambda_{\mathrm{reg}}\|\psi\|_2^2
\right].
$$

先設 $\lambda_E=0$ 訓練有逐次觀測的端點，再以共享參數延續訓練並加入效應損失。第二階段每輪繼續使用逐次標註，避免只配合平均效應而犧牲條件內校準。比較逐次損失與聯合損失時固定相同訓練資料分割、可用特徵和讀出頭。權重在開發資料選定。若加入 LoRA，$\theta=\theta_0+\Delta\theta$ 的選定矩陣以低秩參數更新，輸入、因果遮罩與讀出契約維持相同。每個留出材料與實驗家族不參與任何權重、特徵或門檻設定。 [逐階段訓練方案](training-protocol.md#訓練順序)

### 骨幹微調的第一個對照

以同一逐 token 讀出、監督與材料切分，比較凍結 Qwen 和一組局部 LoRA。第一個可檢驗的介入位置是第 24 層 full attention 的 query 路徑與 `k_proj`。Q/K 直接進入已讀位置的 attention logits。Qwen 的 `q_proj` 在每個 head 的輸出中交錯放置 query 與輸出 gate，query-only LoRA 只更新各 head 的 query 輸出列。原矩陣保持凍結，低秩增量改變有效權重。訓練時可快取第 24 層之前的全序列凍結狀態，再由可微的第 24 層重新計算因果 attention 和讀出依賴的表徵。推論依原契約完成一次前向。 [固定版 Qwen 實作](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L688-L743) · [LoRA 定義](https://arxiv.org/abs/2106.09685)

兩組各自確認回報基準率、讀出收斂及因果前綴，再以未見材料評估位置誤差、校準，以及相同目標詞換前文時的真人效應。先前 PLABA 的微調位於末層 MLP `down_proj`，評估句級自評，該組合的折外 NLL 較高。這個結果約束該組合。各層對未解疑問維持或解開的作用仍需分別證實。 [說明文 LoRA 結果](expository_lora/results.md) · [Frick 基準率診斷](union_training/localization-results.md#frick-讀出在整體停訓時的擬合程度)

## 5. 同一疑問何時維持、解開或重新出現

基本模型的 $p_{t,k}$ 是指定條件下的停點回報比例。$p_{t+1,k}-p_{t,k}$ 反映群體比例變化。同一讀者在兩個位置的轉移，需要該讀者對同一疑問的追蹤。舉例來說，兩個停點各有一半讀者仍有疑問，可以是同一半讀者一直有疑問，也可以是兩批人互換。邊際比例無法區分兩種情況。

有追蹤資料時，可為每個可追蹤的疑問 $r$ 建立離散狀態 $S_{t,r}$。這一節的 $r$ 含可在當時辨識的區段和疑問識別。共用區段的多個疑問須有可對照標籤。候選狀態集合 $\mathcal S=\{N,U,P,R\}$ 分別表示尚未出現、仍未解、暫擱、已確認解開。暫擱需定義可操作的追蹤或提取量測，由資料判定這個狀態是否可用。從原文候選區段可見時開始追蹤，前一位置初始狀態為 $N$。合法轉移集合 $\mathcal A$ 事先定義，例如 $\mathcal A(N)=\{N,U\}$、$\mathcal A(U)=\{U,P,R\}$、$\mathcal A(P)=\{P,U,R\}$、$\mathcal A(R)=\{R,U\}$。

在此模型中，給定 $x_{\le t},c,r$，下一個狀態只再依賴前一個狀態。對合法狀態對 $(a,b)$，以平行讀出頭 $g_{a,b}(\phi_t,u_r)$ 給分數，其中 $u_r$ 包含所指區段與疑問類別的表示。每個起始狀態單獨正規化，得到轉移矩陣

$$
K_{t,r}(a,b)=
\begin{cases}
\exp g_{a,b}(\phi_t,u_r)\big/\displaystyle\sum_{b'\in\mathcal A(a)}
                       \exp g_{a,b'}(\phi_t,u_r), & b\in\mathcal A(a),\\
0, & b\notin\mathcal A(a).
\end{cases}
$$

令 $e$ 為候選區段首次完全可見的位置，$\mu_{e-1,r}\in\mathbb R^{1\times|\mathcal S|}$ 為集中於 $N$ 的初始分布。以下矩陣積依 token 順序從左到右相乘。讀到 $t$ 時的狀態機率與個別轉移事件的機率為

$$
\mu_{t,r}=\mu_{e-1,r}\prod_{s=e}^{t}K_{s,r},\qquad
\Pr(S_{t-1,r}=a,S_{t,r}=b)=\mu_{t-1,r}(a)K_{t,r}(a,b).
$$

例如 $N\to U$ 為首次出現，$U\to R$ 為解開，$P\to U$ 為暫擱後重現，$R\to U$ 為確認解開後再次出現。疑問仍未解且累積的群體量可定義為 $\sum_r\mu_{t,r}(U)$，在候選疑問互相區別的前提下，它是未解疑問的期望個數，單位是「個」。下一句開頭的「接不上」維持專用的 $p_{t,k}$，可在仍未解或正在暫擱等不同情況發生。這項回報與未解疑問個數是不同觀察。

觀察同一疑問在 $u,v$ 的精確狀態 $a,b$ 時，區間的負對數似然為

$$
-\log\left[\prod_{s=u+1}^{v}K_{s,r}\right]_{a,b}.
$$

對多個帶誤差或部分觀測的停點，令 $D_{t,r}$ 為對角矩陣，其對角值為 $P_\psi(o_{t,r}\mid S_{t,r}=a,\phi_t,u_r)$。給定各停點的狀態和特徵，各次觀測依其對應分布產生。有明確狀態標註時可用相符狀態的指示函數，無觀測時取單位矩陣。整段試次似然為

$$
P_\psi(o^{\mathrm{obs}}_r\mid X,c)
=\mu_{e-1,r}\left(\prod_{s=e}^{T}K_{s,r}D_{s,r}\right)\mathbf 1 .
$$

加上追蹤資料後，可以最小化其負對數似然。若改由狀態模型解釋同一個停點端點，則用 $\sum_a\mu_{t,r}(a)P_\psi(o_{t,r}\mid S_{t,r}=a,\phi_t,u_r)$ 導出該端點的回報機率。這樣對同一端點，轉移和停點比例由同一模型決定。

對 TED-Q 類的序位回報，將量尺加入觀測模型 $D_{t,r}$。停止追蹤是由先前已記錄的回報和摘錄末點決定。在原規則下，首次達到 3 分可用來評部分獲答的區間。要評完全獲答時，3–4 分後的未評停點取單位矩陣，原問題可繼續轉移。不可用同一條停止規則把其潛在狀態設為 $R$。以已記錄的 5 分直接驗收時，須保留第一次未評但已有下一段的分母及其界限。 [TED-Q 實測缺失範圍](semantic_tracking/tedq-full-answer-bounds.md)

所有 $K_{t,r}$ 和報告機率均由同一次 Qwen forward 及平行讀出取得。上式的矩陣乘積由損失函數或分析程式計算，也可用保序的 associative scan 同時求所有前綴乘積。這一步沒有將抽樣出的狀態回餵給 Qwen。兩個停點即使同一讀者、同一疑問，通常也只約束兩者間的矩陣乘積。把更新全部放在第一個或最後一個未觀測 token，仍可得到相同的區間似然。若既有資料在區間內缺少更細觀測，驗證只報在該區間。後進先出則需要既有資料可觀測多個疑問的出現與解除順序。

## 6. 驗證、定位與新預測

### 工程驗證

令 $O_{\le t}$ 包含到 $t$ 為止的所有 token 索引讀出，以及其已讀候選指向。對兩篇同背景、同位置編號與可見 token 前綴相同的輸入，

$$
x^A_{\le t}=x^B_{\le t}\ \Longrightarrow\
O_{\le t}(X^A,c)=O_{\le t}(X^B,c).
$$

理論上的等式用確定性推論設定和該後端的數值容差查驗。測「相同 token 前綴換後文」及「全文與截短前綴」兩種比較。構造測試時先核對 tokenizer 在邊界產生的實際 token IDs。若比較字元區段輸出，也核對共用前綴的原文字元偏移。用同樣的 causal policy 檢查所有特徵、候選與正規化。候選不能因為全文排序而在回頭比較前綴時改變。

### 真人效度

同一材料家族的原文、改寫、共享材料與再分析放在同一分割。模型、特徵、權重、門檻及評分規則固定後，報告留出材料、讀者或實驗家族的表現。依不同端點報告：

| 端點 | 驗證量 |
| --- | --- |
| 反應是否出現 | 留出試次的 log loss、Brier score 及分箱校準。Brier score 是預估機率與二元回報的均方差，範圍 $[0,1]$，越低越好。 |
| 反應指向哪裡 | 真人金標的候選涵蓋率，含候選錯失的完整定位分數。以原文字元區段的一對一配對計算 exact match 或事前定義的重疊準則。 |
| 主觀強度 | 序位負對數似然與原量尺上平均分數的絕對誤差。 |
| 何時出現或改變 | 比較預先對齊的真人停點與回報機率，用下式報方向、幅度與可辨認的位置範圍。 |

以材料家族 $d$、版本 $v$、讀者群 $g$、預定對齊停點 $j$ 為索引。令 $\tau(d,v,j)$ 把同一實驗位置映射至各版本的合法 token。以共有的材料和讀者背景分布 $\nu_g$ 彙整兩邊結果。模型估計與條件效應為

$$
\begin{aligned}
\mu^{M}_{v,g,j,k}
 &=\mathbb E_{(d,c)\sim\nu_g}
       [p_{d,v,\tau(d,v,j),k}(c)],\\
\Delta^{M}_{g,j,k}&=\mu^{M}_{B,g,j,k}-\mu^{M}_{A,g,j,k},\\
E_{g,j,k}&=\Delta^{M}_{g,j,k}-\Delta^{H}_{g,j,k}.
\end{aligned}
$$

$\Delta^H$ 從相同量尺和權重的真人試次估計。隨機分派版本時，它估計該群的文字改寫效果。比較不同背景組的 $\Delta$ 可檢查效果異質性。區間估計依研究的分派和抽樣單位處理受試者、材料，以及版本配對。事先設定可接受誤差 $\varepsilon$ 時，可以預定信賴區間須全部落在 $[-\varepsilon,\varepsilon]$ 作為重現判準。

若要估計最早的差異位置，先定義具有實質意義的幅度 $\delta>0$、方向、停點網格及多重比較的區間。模型可在預先指定的方向首次跨過 $\delta$ 時提出位置預測。真人結果只在實際停點檢驗，並分清「第一個有差異證據的停點」與「已有證據約束的出現區間」。若前一點的區間顯示近似無差異，而後一點的區間顯示超過 $\delta$，兩點之間可作為局部化範圍。沒有密集停點時，不能判定中間是否先出現又消失。

## 7. 初版的實驗契約

首輪按各資料的既有觀測定義參考讀者群與端點。用 $\mathcal L_Y$ 訓練有分母的反應類型，以 $\mathcal L_Z$ 訓練有區段標註的讀出，並用序位頭處理有對應的既有量表。狀態轉移僅在有可用且能配對同一讀者、同一疑問的資料上訓練與驗證。觀測只到句或片段時便在該粒度報告，逐 token 反應曲線保留為模型估計。
