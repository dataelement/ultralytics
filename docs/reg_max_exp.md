# row col det
image size: 640
epoch: 10


1. reg_max=1(相当于不使用 DFL)
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       1339      27324      0.736      0.537      0.548      0.334
          table column       1339       8895      0.952      0.933      0.955      0.691
             table row       1339      15997      0.697      0.474      0.453      0.197
   table spanning cell        469       2432      0.558      0.204      0.236      0.116
```

2. reg_max=16(image size / 40)
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       1339      27324      0.861      0.793      0.855      0.694
          table column       1339       8895      0.961      0.966      0.982      0.876
             table row       1339      15997      0.864      0.859       0.91      0.685
   table spanning cell        469       2432      0.758      0.554      0.673      0.522
```

3. reg_max=32(image size / 20)
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       1339      27324      0.943      0.585      0.648      0.452
          table column       1339       8895      0.968      0.883      0.945      0.714
             table row       1339      15997      0.862      0.873      0.928      0.608
   table spanning cell        469       2432          1   0.000515     0.0716      0.034
```

4. reg_max=64(image size / 10)
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       1339      27324      0.945      0.563      0.614      0.461
          table column       1339       8895      0.966       0.82      0.899      0.668
             table row       1339      15997       0.87       0.87      0.933       0.71
   table spanning cell        469       2432          1          0    0.00978     0.0045
```

5. reg_max=128(image size / 5)
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all       1339      27324      0.939      0.528      0.592      0.434
          table column       1339       8895      0.974      0.753      0.867      0.636
             table row       1339      15997      0.843       0.83      0.903      0.663
   table spanning cell        469       2432          1          0    0.00668    0.00333
```