 

## Neural Architechures for Chinese Named Entity Recognition



- ### Architechures

  ![](<https://github.com/zhentaoCoding/ChineseNER/blob/master/img/architecture.PNG>)

- ### word embedding

  glove

- ## Bi-LSTM Layers

  ![](<https://github.com/zhentaoCoding/ChineseNER/blob/master/img/4.PNG>)

- ### CRF Tagging Models

  ![](<https://github.com/zhentaoCoding/ChineseNER/blob/master/img/1.PNG>)

  ![](<https://github.com/zhentaoCoding/ChineseNER/blob/master/img/2.PNG>)

  ![](<https://github.com/zhentaoCoding/ChineseNER/blob/master/img/3.PNG>)

- ### Hper-parameters  Settings

  embedding size : 300

  hidden size : 200

  dropout : 0.5

- ### Result

  |   Model    |     Variant      |  F1  |
  | :--------: | :--------------: | :--: |
  | BiLSTM+CRF | pretrain+dropout | 87.3 |

