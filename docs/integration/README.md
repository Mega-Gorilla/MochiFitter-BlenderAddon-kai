# 統合ドキュメント

Blender アドオンと Unity アドオン間の連携に関するドキュメントです。

## コンテンツ

| ドキュメント | 説明 |
|-------------|------|
| [coordinate_systems.md](coordinate_systems.md) | 座標系とデータ変換（Blender Z-up <-> Unity Y-up） |
| [data_flow.md](data_flow.md) | MochiFitter 全体のデータフロー |

## 概要

MochiFitter は以下のワークフローでアバター衣装のリターゲットを行います：

```
[Phase 1] Blender アドオン
    │
    │  posediff_*.json, deformation_*.npz, avatar_data_*.json
    ▼
[Phase 2] Unity アドオン（設定）
    │
    │  config_*.json
    ▼
[Phase 3] Blender subprocess（リターゲット処理）
    │
    │  output.fbx
    ▼
[Phase 4] Unity（Prefab 生成）
```

## 関連

- [Blender アドオンドキュメント](../blender-addon/)
- [Unity アドオンドキュメント](../unity-addon/)
