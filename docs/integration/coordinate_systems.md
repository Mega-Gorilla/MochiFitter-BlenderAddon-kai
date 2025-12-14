# 座標系とデータ変換

## 概要

MochiFitter は Blender と Unity 間でデータをやり取りするため、
異なる座標系間の変換を適切に処理する必要があります。

## 座標系の違い

### Blender

```
      Z (上)
      │
      │
      │
      └────── Y (奥)
     /
    /
   X (右)
```

- **上方向**: Z軸正方向（Z-up）
- **前方向**: Y軸負方向（-Y forward）
- **右方向**: X軸正方向
- **座標系**: 右手系（Right-handed）

### Unity

```
      Y (上)
      │
      │
      │
      └────── X (右)
     /
    /
   Z (前)
```

- **上方向**: Y軸正方向（Y-up）
- **前方向**: Z軸正方向（Z forward）
- **右方向**: X軸正方向
- **座標系**: 左手系（Left-handed）

## 変換の仕組み

### FBX エクスポート時の変換

リターゲットスクリプトの FBX エクスポート設定：

```python
bpy.ops.export_scene.fbx(
    filepath=filepath,
    use_selection=selected_only,
    apply_scale_options='FBX_SCALE_ALL',
    apply_unit_scale=True,
    add_leaf_bones=False,
    axis_forward='-Z',  # Blender の -Y を Unity の -Z に
    axis_up='Y'         # Blender の Z を Unity の Y に
)
```

この設定により、Blender の座標系が自動的に Unity の座標系に変換されます。

### 座標軸の対応

FBX エクスポート設定（`axis_forward='-Z'`, `axis_up='Y'`）により、
以下の対応で座標変換が行われます：

| Blender | Unity |
|---------|-------|
| X | X |
| Y | -Z |
| Z | Y |

> **Note**: 右手系から左手系への変換を含むため、単純な軸入れ替えではなく
> 符号反転も含まれます。FBX エクスポーターがこの変換を自動的に行うため、
> 通常は手動での変換は不要です。

## MochiFitter でのデータ処理

### 内部処理（Blender 座標系）

MochiFitter の内部処理はすべて Blender の座標系で行われます：

1. **ポーズデータ（posediff_*.json）**
   - `delta_matrix`: Blender のワールド座標系での 4x4 変換行列
   - `location`: Blender 座標系での移動量 (X, Y, Z)
   - `rotation`: オイラー角（度）、XYZ 順
   - `head_world`: Blender ワールド座標

2. **変形フィールド（deformation_*.npz）**
   - `all_field_points`: Blender ワールド座標
   - `all_delta_positions`: Blender 座標系での変位ベクトル
   - `world_matrix`: 現在は単位行列（将来の拡張用）

3. **アバターデータ（avatar_data_*.json）**
   - ボーン階層情報（座標系に依存しない）
   - Humanoid ボーンマッピング

### FBX エクスポート時

リターゲット処理後、FBX エクスポート時に座標変換が適用されます：

```
┌─────────────────────────────────────────────────────────────┐
│                    座標変換フロー                            │
└─────────────────────────────────────────────────────────────┘

[Blender 内部処理]          [FBX エクスポート]         [Unity]
     │                           │                      │
  Z-up                    座標系変換                  Y-up
  右手系          ───────────────────────────>       左手系
     │              axis_forward='-Z'                  │
     │              axis_up='Y'                        │
     │                                                 │
posediff.json                                   FBX インポート
deformation.npz                                 自動変換済み
```

## 実装上の注意点

### world_matrix の制限

現在、NPZ ファイルの `world_matrix` は常に単位行列です：

```python
world_matrix = np.identity(4)
```

これは以下の前提に基づいています：
- 変形フィールドの計算はワールド座標系で行われる
- **前提**: ソースオブジェクトの変換が適用済み（Apply Transform）であること
  - 未適用の場合、変形結果が正しくない可能性があります
- 将来の拡張のために保存されている

### ボーン行列の処理

ポーズ適用時のボーン行列変換：

```python
# ワールド空間での行列計算
world_matrix = armature_obj.matrix_world @ bone.matrix

# 差分行列の計算
delta_matrix = world_matrix @ base_world_matrix.inverted()

# ポーズ適用時
combined_matrix = delta_matrix @ current_world_matrix
bone.matrix = armature_obj.matrix_world.inverted() @ combined_matrix
```

すべての計算は Blender のワールド座標系で行われ、
最終的な FBX エクスポート時にのみ座標変換が適用されます。

## トラブルシューティング

### メッシュが反転している場合

FBX インポート/エクスポート設定を確認：
- Unity 側のインポート設定で軸が正しいか確認
- Scale Factor が適切か確認（通常は 1.0）

### ボーンの向きがおかしい場合

- FBX エクスポート時の `axis_forward` と `axis_up` 設定を確認
- Unity の Rig 設定で「Optimize Game Objects」が有効になっていないか確認

### スケールが合わない場合

- Blender のユニット設定を確認（メートル単位推奨）
- FBX エクスポートの `apply_scale_options` を確認
- Unity のインポート設定で Scale Factor を確認

## 関連ドキュメント

- [データフロー](data_flow.md)
- [Blender アドオン - データフォーマット](../blender-addon/data_formats.md)
- [Unity アドオン - リターゲットスクリプト](../unity-addon/blender_script.md)
