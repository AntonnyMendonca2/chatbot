
  <blockquote style="background-color: #fff000 ; padding: 10px; border-left: 5px solid #000000; margin: 0;">
    <p><strong>Atenção:</strong> Assista ao vídeo abaixo para um melhor entendimento.</p>
</blockquote>
<br>

![exeplo postman requisicao](/imagens/video.png)


Basta clicar aqui [aqui](https://youtu.be/h29XJOFVHEc?si=wz28c4q8bKtJkBCy) para ser redirecionado ao vídeo gravado por mim explicando sobre a estrutura e o funcionamento da aplicação, é de extrema importância que assista.


# Estrutura de Pastas do Projeto

- 📁 **PROCESSO_SELETIVO**
  - 📁 **chatbot**
    - 📄 train.py (Algoritmo para treinamento do modelo)
    - 📄 intents.json (O arquivo com inteções para o chatbot identificar)
    - 📄 chatbot_neural.py (O arquivo que deve ser rodado para consumir/disponibilizar o bot para API)
  - 📁 **imagens**
  - 📁 **data engineering**
    - 📁 data
      - 📄 clietes.csv
      - 📄 clietes.json
      - 📄 vendas.csv
      - 📄 vendas.json
    - 📄 config.json (Arquivo com as credenciais para acessar tanto o mysql quanto o postgres)
    - 📄 connectionn.py (Arquivo que cria a conexão com o banco de dados mysql)
    - 📄 data_engineering.ipynb (Arquivo .ipynb onde eu faço a engenharia de dados)
  - 📄 README.md
  - 📄 requirements.txt (bibliotecas necessárias para o bom funcionamento do código)

<br>

  <blockquote style="background-color: #ffcccc; padding: 10px; border-left: 5px solid #ff0000; margin: 0;">
    <p><strong>Atenção:</strong> Para o funcionamento do código, as dependências devem ser instaladas dentro de um ambiente virtual, segue a sequência de comandos que devem ser digitados no console.</p>
</blockquote>

<BR>
Digite o código abaixo linha por linha no console do visual studio code:

<BR>

```bash
    python -m venv environment
    .\environment\Scripts\activate
    pip install -r requirements.txt
```

para o treinamento do modelo, basta entrar na pasta e rodar o arquivo train.py:

```bash
    cd chatbot
    python train.py
```

para disponibilizar o bot para ser consumido através de uma API, basta rodar o arquivo chatbot_neural.py:

```bash
    python chatbot_neural.py
```

Para mandar requisições à aplicação, utilizar um software como Postman e enviar uma requisição POST no link abaixo:

```
    http://127.0.0.1:5000/receive_message
```

não esqueça de especificar no headers a chave > valor:

![exeplo postman](/imagens/postman.png)

envie no corpo da requisição um json com a mensagem que desejar, assim como em uma conversa do whatsapp:

![exeplo postman requisicao](/imagens/exemplo_requisicao.png)
