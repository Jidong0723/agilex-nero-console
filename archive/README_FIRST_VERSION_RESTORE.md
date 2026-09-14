# 第一版本回档

第一版原始压缩包的固定 SHA-256：

`641F72BFCE1E523771E68C13A3E786CE5A52424125DE755C15E11E9E4A1CECB0`

默认恢复命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restore_first_version.ps1
```

默认情况下，脚本会在当前项目的上级目录查找
`agilex-nero-console.zip`，校验后恢复到同级的
`restored\agilex-nero-console-first-version`。
它拒绝覆盖已经存在的目标目录，也不会修改当前运行目录。

如需指定另一空目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restore_first_version.ps1 -ArchivePath D:\archives\agilex-nero-console.zip -Destination D:\restored\my-first-version
```
