import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["PoolFormer","poolformer_s12","poolformer_s24","poolformer_s36","poolformer_m36","poolformer_m48"]


trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class PatchEmbed(nn.Layer):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride](do not reshape to [B,num_patches,C], because we will use pooling operator)
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x # [B, C, H/stride, W/stride]

class LayerNormChannel(nn.Layer):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, epsilon=1e-05):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=ones_)
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=zeros_)
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
 
### Pooling        
class Pooling(nn.Layer):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool2D(
            kernel_size, stride=1, padding=kernel_size//2, exclusive=True)


    def forward(self, x):
        return self.pool(x) - x # 

### MLP
class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.act(x)     
        x = self.drop(x)
        x = self.fc2(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.drop(x)
        return x            
    
### PoolFormerBlock
class PoolFormerBlock(nn.Layer):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0.,  
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(kernel_size=pool_size)   
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:

            self.layer_scale_1 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

            self.layer_scale_2 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale: 
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) # [dim,1,1]*[B,dim,H,W]--->[B,dim,H,W]
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x # <B,C,H,W>

### Basic Block   
def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks

def poolformer_s12(**kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    return model

### PoolFormer:remove head ,then use this for DIY model
class PoolFormer(nn.Layer):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    """
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 use_attention=True,attention_drop_rate=0.,
                 activations = True,
                 **kwargs):

        super().__init__()
        # whether activations
        self.activations = activations
        self.use_attention = use_attention
        # 定义 patch embed
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])
        
        # define transformer block # TODO
        self.transformer_block = Transformer_Encoder(
            embed_dim=embed_dims[3],
            num_heads=8,
            depth=layers[3],
            attn_head_size=None,
            qkv_bias=True,
            mlp_ratio=4.0,
            dropout=drop_rate,
            attention_dropout=attention_drop_rate,
            droppath=drop_path_rate
        )
        
        # add a 1*1 conv at end # TODO
        self.final_mlp = nn.Conv2D(embed_dims[3],embed_dims[3],1)
        

        # set the main block in network
        network = []
        if self.use_attention: # TODO
            # [B,embed_dims[3],H/32,W/32]
            for i in range(len(layers)-1): 
                stage = basic_blocks(embed_dims[i], i, layers, 
                                    pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                    act_layer=act_layer, norm_layer=norm_layer, 
                                    drop_rate=drop_rate, 
                                    drop_path_rate=drop_path_rate,
                                    use_layer_scale=use_layer_scale, 
                                    layer_scale_init_value=layer_scale_init_value)
                network.append(stage)
                # if i >= len(layers) - 1: 
                #     break
                if downsamples[i] or embed_dims[i] != embed_dims[i+1]: 
                    # downsampling between two stages
                    network.append(
                        PatchEmbed(
                            patch_size=down_patch_size, stride=down_stride, 
                            padding=down_pad, 
                            in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                            )
                        )
            
            self.network = nn.LayerList(network)
            
        
        else:        
            for i in range(len(layers)):
                stage = basic_blocks(embed_dims[i], i, layers, 
                                    pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                    act_layer=act_layer, norm_layer=norm_layer, 
                                    drop_rate=drop_rate, 
                                    drop_path_rate=drop_path_rate,
                                    use_layer_scale=use_layer_scale, 
                                    layer_scale_init_value=layer_scale_init_value)
                network.append(stage)
                if i >= len(layers) - 1: 
                    break
                if downsamples[i] or embed_dims[i] != embed_dims[i+1]: 
                    # downsampling between two stages
                    network.append(
                        PatchEmbed(
                            patch_size=down_patch_size, stride=down_stride, 
                            padding=down_pad, 
                            in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                            )
                        )

            self.network = nn.LayerList(network)

        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else Identity()

        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x) # [N,C,7,7]
        if self.use_attention: # TODO
            # reshape：[B,C,H,W]--->[B,num_tokens,C]
            b,c,h,w = x.shape
            x = x.flatten(2)  # [B, C, H, W] -> [B, C, h*w]
            x = x.transpose([0, 2, 1])  # [B, C, h*w] -> [B, h*w, C] = [B, N, C]            
            # transformer block
            x = self.transformer_block(x) # [B,h*w,C]
            # reshape:[B,h*w,C]--->[B,C,H,W]
            x = x.transpose([0,2,1]) # [B,C,h*w]
            x = x.reshape([b,c,h,w])
            x = self.final_mlp(x) # TODO
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        x = self.norm(x)
        activations = x 
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        if self.activations:
            return activations, cls_out
        else:
            return cls_out 


def poolformer_s12(**kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    return model

def poolformer_s24(**kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    return model

def poolformer_s36(**kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model


def poolformer_m36(**kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model


def poolformer_m48(**kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        layer_scale_init_value=1e-6, 
        **kwargs)
    return model

#### Attention Block Code

class Attention(nn.Layer):
    """ Attention module
    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        num_heads: number of heads
        attn_head_size: feature dim of single head
        all_head_size: feature dim of all heads
        qkv: a nn.Linear for q, k, v mapping
        scales: 1 / sqrt(single_head_feature_dim)
        out: projection of multi-head attention
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if attn_head_size is not None:
            self.attn_head_size = attn_head_size
        else:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3,  # weights for q, k, and v
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(self.all_head_size,
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        """[B, N, C] -> [B, N, n_heads, head_dim] -> [B, n_heads, N, head_dim]"""
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)  # [B, N, C] -> [B, N, n_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim] -> [B, n_heads, N, head_dim]
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)  # [B, n_heads, N, N]
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)  # [B, n_heads, N, head_dim]
        z = z.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim]
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)  # [B, N, all_head_size]

        z = self.out(z)
        z = self.proj_dropout(z)
        return z # [B, N, all_head_size]


class Mlp2(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc
    """

    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=0.2))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Layer):
    """Transformer Layer
    Transformer layer contains attention, norm, mlp and residual
    Attributes:
        embed_dim: transformer feature dim
        attn_norm: nn.LayerNorm before attention
        mlp_norm: nn.LayerNorm before mlp
        mlp: mlp modual
        attn: attention modual
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.attn_norm = nn.LayerNorm(embed_dim,
                                      weight_attr=w_attr_1,
                                      bias_attr=b_attr_1,
                                      epsilon=1e-6)

        self.attn = Attention(embed_dim,
                              num_heads,
                              attn_head_size,
                              qkv_bias,
                              dropout,
                              attention_dropout)

        #self.drop_path = DropPath(droppath) if droppath > 0. else Identity()

        w_attr_2, b_attr_2 = self._init_weights()
        self.mlp_norm = nn.LayerNorm(embed_dim,
                                     weight_attr=w_attr_2,
                                     bias_attr=b_attr_2,
                                     epsilon=1e-6)

        self.mlp = Mlp2(embed_dim, mlp_ratio, dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        #x = self.drop_path(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        #x = self.drop_path(x)
        x = x + h

        return x


class Transformer_Encoder(nn.Layer):
    """Transformer encoder
    Encoder encoder contains a list of TransformerLayer, and a LayerNorm.
    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        # stochatic depth decay
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(TransformerLayer(embed_dim,
                                               num_heads,
                                               attn_head_size,
                                               qkv_bias,
                                               mlp_ratio,
                                               dropout,
                                               attention_dropout,
                                               depth_decay[i]))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights()
        self.encoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return x # [B,N,all_head_dim]
    
if __name__ == '__main__':
    poolformer_m48_origin = poolformer_m48(use_attention=False)
    poolformer_m48_attention = poolformer_m48(use_attention=True)
    input = paddle.rand([4,3,224,224])
    # output_poolformer_m48_origin = poolformer_m48_origin(input)
    # output_poolformer_m48_attention = poolformer_m48_attention(input)
    # print(output_poolformer_m48_attention.shape)
    # print(output_poolformer_m48_origin.shape)
    param_info = paddle.summary(poolformer_m48_origin,(1,3,224,224))
    print(param_info)
    