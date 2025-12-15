# profile_data

検証・テスト用のデータファイル格納フォルダです。

> **Note**: このフォルダは `.gitignore` で除外されており、Git にはコミットされません。
> 有料アバターの FBX や著作権のあるファイルを含む可能性があるためです。

## フォルダ構造

```
profile_data/
├── README.md                              # このファイル
│
├── # ===== Avatar Data =====
├── avatar_data_beryl4.json                # Beryl アバター情報
├── avatar_data_template.json              # Template アバター情報
├── avatar_data_mao.json                   # mao アバター情報
│
├── # ===== Pose Data =====
├── pose_basis_beryl.json                  # Beryl ベースポーズ
├── pose_basis_template.json               # Template ベースポーズ
├── pose_basis_mao.json                    # mao ベースポーズ
│
├── # ===== Pose Diff (変換差分) =====
├── posediff_beryl_to_template.json        # Beryl → Template
├── posediff_template_to_beryl.json        # Template → Beryl
├── posediff_template_to_mao.json          # Template → mao
│
├── # ===== Deformation Fields =====
├── deformation_beryl_to_template.npz      # Beryl → Template 変形フィールド
├── deformation_template_to_mao.npz        # Template → mao 変形フィールド
├── deformation_*.npz                      # その他の変形フィールド
│
├── # ===== Config Files =====
├── config_beryl2template.json             # Beryl → Template 設定
├── config_template2mao.json               # Template → mao 設定
│
└── fbx/                                   # FBX ファイル
    ├── Beryl.fbx                          # Beryl アバター
    ├── Beryl_Costumes.fbx                 # Beryl 衣装
    ├── Template.fbx                       # Template アバター
    └── mao.fbx                            # mao アバター
```

## 用途

### Issue #15 の検証

チェーン処理（Beryl → Template → mao）の問題を検証するために使用：

```bash
# リターゲット実行例
blender --background --python retarget_script2_12.py -- \
    --input profile_data/fbx/Beryl_Costumes.fbx \
    --output output.fbx \
    --base-fbx "profile_data/fbx/Template.fbx;profile_data/fbx/mao.fbx" \
    --config "profile_data/config_beryl2template.json;profile_data/config_template2mao.json"
```

### ファイルの入手方法

1. **Beryl**: 有料アバター（BOOTH等で購入）
2. **Template**: MochiFitter Unity アドオンに付属
3. **mao**: 有料アバター（BOOTH等で購入）

## 関連

- [Issue #15: チェーン処理時のHips位置・スケール調整が適用されない](https://github.com/Mega-Gorilla/MochiFitter-BlenderAddon-kai/issues/15)
- [BlenderScripts](../MochFitter-unity-addon/BlenderScripts/)
