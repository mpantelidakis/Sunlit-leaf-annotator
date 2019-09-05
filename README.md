# Título: Pynovisao
## Autores (ordem alfabética): Adair da Silva Oliveira Junior, Alessandro dos Santos Ferreira, Diego André Sant'Ana(diegoandresantana@gmail.com), Diogo Nunes Gonçalves(dnunesgoncalves@gmail.com), Everton Castelão Tetila(evertontetila@gmail.com), Felipe Silveira(eng.fe.silveira@gmail.com), Gabriel Kirsten Menezes(gabriel.kirsten@hotmail.com), Gilberto Astolfi(gilbertoastolfi@gmail.com), Hemerson Pistori (pistori@ucdb.br), Nícolas Alessandro de Souza Belete(nicolas.belete@gmail.com)

## Resumo:

Pacote de Visão Computacional do Inovisão.

## Licença de Uso: 

NPOSL-30 https://opensource.org/licenses/NPOSL-3.0 - Livre para uso apenas sem fins lucrativos (E.g.: ensino, pesquisa científica, etc). Entrar em contato com o coordenador do grupo Inovisão, Prof. Hemerson Pistori (pistori@ucdb.br), caso tenha interesse na exploração comercial do software.


## Como citar:

[1] dos Santos Ferreira, A., Freitas, D. M., da Silva, G. G., Pistori, H., & Folhes, M. T. (2017). Weed detection in soybean crops using ConvNets. Computers and Electronics in Agriculture, 143, 314-324.

## Como Usar

- A partir da pasta raiz, execute os seguintes comandos:

```
 $ cd src
```

```
 $ python main.py
```

- Uma imagem como a mostrada abaixo deve ser apresentada:

    ![pynovisao](data/pynovisao.png)
    
## Outras Opções

- Mostra todas as opções disponíveis

```
 $ python main.py --help
```
- Executa o programa inicializando o banco de imagens em *../data/soja*

```
 $ python main.py --dataset ../data/soja
```
- Executa o programa definindo as classes e suas respectivas cores (X11 color names)

```
 $ python main.py --classes "Solo Soja Gramineas FolhasLargas" --colors "Orange SpringGreen RebeccaPurple Snow"
```

- Existe também um script para Linux que ajuda a dividir o conjunto de imagens entre treinamento, validação e teste. Ainda não está integrado à interface. Para saber mais: 
```
 $ cd src/util
 $ chmod 755 split_data.sh
 $ ./split_data -h
```


## Como instalar (opção 1, somente linux)
### Linux
Você pode instalar utilizando o script de instalação realizando os seguintes passos:

- Na pasta raiz do projeto, execute o comando abaixo para ceder a permissão de execução no script de instalação.
```
$ sudo chmod a+x INSTALL.sh
``` 
 
- Execute o script de instalação.
```
$ sudo ./INSTALL.sh
```

O script de instalação foi testado na versão 16.04 do Ubuntu.


## Como instalar (opção 2, sem o script)

### Dependências
#### Linux

Será necessário instalar:
- Python 2.7.6
- Opencv 2.7
- tk/tk-dev

As bibliotecas necessárias podem ser encontradas no arquivo __requeriments.txt__ na raiz do projeto, utilize o comando __pip__ para instalar.
- Instalação do pip:
```
$ sudo apt-get install python-pip
```
- Instale o numpy:
```
$ sudo pip install numpy
```
- Instação das bibliotecas:
```
$ sudo pip install -r requeriments.txt
```

### Windows

- Instale o [Anaconda](http://continuum.io/downloads) que contém todas dependências, inclusive o Python. Basta fazer o download do arquivo .exe e executá-lo.
- Opencv 2.7
- python-weka-wrapper ( Classification )

### Como instalar o OpenCV 

#### Linux (caso pip não funcione)

Seguir as instruções disponíveis em [OpenCV-Linux](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation). Lí em algum lugar que dá para instalar com o comando abaixo, não testei mas pode funcionar:
```
 $ sudo apt-get install python-opencv
```

Pode ser que seja necessário instalar também uma versão mais antiga do opencv (2.4*) caso apareça um erro com o comando import cv (que sumiu na versão 3.0.0 do opencv). Neste caso, tente seguir estes passos: [Instalando opencv 2.4] (https://sites.google.com/a/computacao.ufcg.edu.br/lvc/aprendizado/opencv).

#### Windows

 - [OpenCV-Python](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows).
	1. Baixe o [Opencv](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows)
	2. Extraia os arquivos no local desejado.
	3. Vá para a pasta opencv/build/python/2.7.
	4. Cipie o arquivo cv2.pyd para C:/Python27/lib/site-packeges.
	5. Abra o terminal e digite python para executar o interpretador.
	6. Digite:
    	
      ```
        >>> import cv2
        >>> print cv2.__version__
      ```

### Como instalar scikit-image e arff (caso pip não funcione)
```
 $ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose python-pip python-networkx libfreetype6-dev

 $ sudo pip install -U scikit-image
```

Em uma das máquinas em que tentei instalar deu um erro que resolvi rodando o comando abaixo antes de executar a linha acima:
```
 $ sudo apt-get build-dep python-matplotlib
 $ sudo pip install cycler
```

### Como instalar o tk/tk-dev

#### Ubuntu 

```
 $ sudo apt-get install tk tk-dev
```
    
Na ocorrência do erro 'cannot import name _tkagg', tentar os seguintes comandos:
    
```
 $ sudo apt-get install tk tk-dev
 $ sudo pip uninstall matplotlib
 $ sudo pip install matplotlib
```

Se der erro na reinstalação do matplotlib (depois que desinstalar), tente desinstalar também pelo apt-get:
```
 $ sudo apt-get remove python-matplotlib
```

### Mais informações
    
- http://www.tkdocs.com/tutorial/install.html


### Como instalar o python-weka-wrapper ( Opcional )

#### Ubuntu (caso pip não funcione) 

Primeiro você precisa compilar os código C/C+ e os módulos Python:
```
$ sudo apt-get install build-essential python-dev
```

Agora você pode instalar os vários pacotes que precisamos para instalar o python-weka-wrapper:
```
$ sudo apt-get install python-pip python-numpy
```

Os seguintes pacotes são opcionais mas necessários se você deseja uma representação gráfica:
```
$ sudo apt-get install python-imaging python-matplotlib python-pygraphviz
```

Instale OpenJDK para obter todos os cabeçalhos que javabridge compila:
```
$ sudo apt-get install default-jdk
```

No meu ubuntu 14.04 tive problemas com dependência, acabei instalando o java da oracle seguindo as orientações deste site: [instalando java da oracle](http://askubuntu.com/questions/521145/how-to-install-oracle-java-on-ubuntu-14-04)

Finalmente você pode usar pip para instalar os pacotes Python que não estão disponíveis no repositório:
```
$ sudo pip install javabridge
$ sudo pip install python-weka-wrapper
```
    
#### Windows

Por favor note: você precisa certificar-se que os bits do seu ambiente é consistente. Isto é, se você instalar uma versão de Python 32-bit você deve instalar um JDK 32-bit e numpy 32-bit ( ou então todos eles devem ser 64-bit ).

Realize os seguintes passos:

Instale Python, esteja certo que você checou Add python.exe to path durante a instalação.

Adicione os scripts Python eu sua variável de ambiente PATH, por exemplo, :\\Python27\\Scripts

Instale pip com os seguintes passos:
- baixe daqui https://bootstrap.pypa.io/get-pip.py
- instale usando python get-pip.py

Instale numpy
- baixe numpy 1.9.x MKL ( ou superior ) para Python 2.7 (cp27) e sua configuração de bit  (32 ou 64 bit)
- instale o arquivo .whl usando pip: pip install numpy-X.Y.Z.whl

Instale .Net 4.0 (se já não estiver instalado)

Instale Windows SDK 7.1

Abra o prompt de comando do Windows SDK (não o prompt de comando convencional!) e instale javabridge e python-weka-wrapper
```
> set MSSdk=1
> set DISTUTILS_USE_SDK=1
> pip install javabridge
> pip install python-weka-wrapper
```

Agora você pode executar python-weka-wrapper usando o prompt de comando convencional também.

Se você deseja as funcionalidades gráficas você precisa instalar matplotlib também:
- baixe matplotlib para Python 2.7 (cp27) e sua configuração de bit (32 or 64 bit)
- instale o arquivo .whl usando pip: pip install matplotlib-X.Y.Z.whl
    
#### Mais informações

- http://pythonhosted.org/python-weka-wrapper/install.html
- http://pythonhosted.org/python-weka-wrapper/troubleshooting.html



### Como instalar o caffe ( Opcional )

#### Ubuntu / Windows

Para poder utilizar o classificador CNNCaffe, uma ConvNet baseada na topologia AlexNet, é necessário instalar o software Caffe.

A instalação do software Caffe é mais complexa que as instalações descritas anteriormente e pode ser encontrada detalhada no link abaixo:
-  http://caffe.berkeleyvision.org/installation.html

Após realizar a instalação do software Caffe, para realizar a classificação, você precisa realizar o treinamento da sua rede no software, pois não há interface no Pynovisao para o treinamento da ConvNet.

O tutorial para o treinamento pode ser encontrado no link abaixo:
- http://caffe.berkeleyvision.org/gathered/examples/imagenet.html

Por fim será necessário configurar sua CNNCaffe.
- Para os campos ModelDef, ModelWeights e MeanImage, você deverá fornecer os caminhos relativos ao seu treinamento realizado no passo anterior.
- Para o campo LabelsFile você deve fornecer o caminho de um arquivo que descrava nominalmente as classes na ordem 0, 1, ..., n-1, onde n é o número de classes que você treinou. 
- Um arquivo de exemplo pode ser encontrado em examples/labels.txt.


# Implementando um novo classificador no pynovisao

Nesta seção será usado como exemplo o classificador **Syntactic**, do tipo **KTESTABLE** e, como opção de hiperparâmetros, o tamanho do vocabulário. 

Inicialmente, você deve criar uma classe onde estão, em um dicionário (chave, valor), todos os tipos do seu classificador. A classe deve ser criada no diretório src/classification/. Veja como exemplo a classe SyntacticAlias no arquivo src/classification/syntactic_alias.py.  

O próximo passo é criar o arquivo (.py) do seu classificador no diretório src/classification/, por exemplo, syntactic.py. No arquivo recém-criado você deve implementar a classe de seu classificador estendendo a classe **Classifier**, que está implementada no arquivo src/classification/classifier.py. Veja exemplo abaixo.

```python
#syntactic.py
#importações mínimas necessárias
from collections import OrderedDict
from util.config import Config
from util.utils import TimeUtils
from classifier import Classifier

class Syntactic(Classifier):
    """Class for syntactic classifier"""
```

No construtor da classe você deve informar valores padrão para os parâmetros. No caso do exemplo abaixo, **classename** é o tipo do classificador e **options** é o tamanho do alfabeto. Além disso, alguns atributos devem ser inicializados:  **self.classname** e **self.options**. O atributo **self.dataset** (opicional) é o path do conjunto de treinamento e teste que o usuário informa na interface gráfica. Ter esse atributo na classe é importante para ter acesso ao conjunto de imagens em qualquer um dos métodos, ele é inicializado no método **train** discutido posteriormente.

```python
def __init__(self, classname="KTESTABLE", options='32'):

        self.classname = Config("ClassName", classname, str)
        self.options = Config("Options", options, str)
        self.dataset = None
        self.reset()
```

O métodos **get_name**, **get_config**, **set_config**, **get_summary_config** e **must_train** possuem implementações padrão. Veja exemplo de implementação em  src/classification/classifier.py.

O método train deve ser implementado para treinar o seu classificador. No parâmetro dataset é passado o diretório onde estão as imagens para treinamento. No corpo do método o valor do atributo self.dataset, declarado como opcional no construtor, é alterado com o atual diretório de treinamento.  

```python
def train(self, dataset, training_data, force = False):              
        
        dataset += '/'
        # atributo que mantém o diretório do dataset.
        self.dataset = dataset 
  	    # os dois testes abaixo são padrão
        
        if self.data is not None and not force:
            return 
        
        if self.data is not None:
            self.reset()
		
	   # implemente aqui seu treinamento.
```

O método **classify** deve ser implementado para que seu classificador faça a predição. No parâmetro **dataset** é passado o diretório onde estão as imagens para treinamento, no **test_dir** é passado a pasta temporária criada pelo pynovisao onde estão as imagens que serão usadas para o teste. A pasta temporária é criada dentro do diretório dataset. Então, para acessar o diretório de testes basta concatenar  **dataset** e **test_dir**, como no exemplo no corpo do método abaixo. O parâmetro test_data é um arquivo arff que contém os dados para os testes. 
 
O método **classify** deve retornar uma lista contendo as classes preditas pelo seu classificador. Exemplo: [‘daninha’,’daninha’,’mancha_alvo’, ‘daninha’]

```python
def classify(self, dataset, test_dir, test_data):
      
	   # diretório onde estão as imagens para testes.
       path_test = dataset + '/' + test_dir + '/'        
        
       # implemente aqui o preditor de seu classificador
 
       return # uma lista com as classes preditas
```

O método **cross_validate** deve ser implementado e o objetivo é implementar a validação cruzada. O método retorna uma string (info) com as métricas. Obs: o atributo **self.dataset** atualizado no método **train** pode ser utilizado no método cross-validate para acessar o diretório das imagens de treinamento. 

```python
def cross_validate(self, detail = True):
        start_time = TimeUtils.get_time()        
        info =  "Scheme:\t%s %s\n" % (str(self.classifier.classname) , "".join([str(option) for option in self.classifier.options]))
	  
	   # implemente aqui a validação cruzada.
	   return info
```
O método **reset** deve ser implementado de forma padrão, como exemplo abaixo.

```python
def reset(self):
        self.data = None
        self.classifier = None
```

Após a implementação de seu classificador, você deve configurá-lo no pynovisao. A configuração deve ser realizada no arquivo src/classification/__init__.py.

Caso você necessite de classes utilitárias, os arquivos delas devem ser criados no diretório  src/util/. Além disso, as classes utilitárias devem ser registradas como módulos no arquivo src/util/__init__.py


Caso dê problema relacionado ao número de processos, adicione as duas váriaveis de ambiente,
sendo que deve adicionar no número de threads que o seu processador permite:

export OMP_NUM_THREADS=8
export KMP_AFFINITY="verbose,explicit,proclist=[0,3,5,9,12,15,18,21],granularity=core"
