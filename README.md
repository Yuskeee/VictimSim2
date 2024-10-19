# VictimSim2

Um simulador projetado para testar algoritmos de busca e outras técnicas de IA em cenários de resgate é utilizado no curso de Inteligência Artificial da UTFPR, em Curitiba. Conhecido como VictimSim2, esse simulador é útil para estudar cenários catastróficos dentro de um ambiente de grade 2D, onde agentes artificiais embarcam em missões de busca e resgate para localizar e ajudar vítimas.

## Principais características do simulador

- O ambiente é composto por uma grade 2D, indexada por coordenadas (coluna, linha) ou (x, y). A origem está situada no canto superior esquerdo, com o eixo y se estendendo para baixo e o eixo x se estendendo para a direita. Enquanto as coordenadas absolutas são acessíveis somente ao simulador de ambiente, os usuários são incentivados a estabelecer seu próprio sistema de coordenadas para os agentes.
- Cada célula dentro da grade 2D é atribuída um grau de dificuldade para acessibilidade, variando de valores superiores a zero até 100. O valor máximo de 100 indica a presença de uma parede intransponível, enquanto valores mais altos significam acesso cada vez mais desafiador. Por outro lado, valores menores ou iguais a um denotam entrada mais fácil.
- O ambiente acomoda um ou mais agentes, com cada agente atribuído uma cor personalizável através de arquivos de configuração.
- A detecção de colisão está integrada para identificar instâncias em que um agente colide com paredes ou atinge os limites da grade, denominado percepção "BUMPED".
- Os agentes possuem a capacidade de detectar obstáculos e limites da grade em seu imediato entorno, um passo à frente de sua posição atual.
- Vários agentes podem ocupar a mesma célula simultaneamente sem causar colisões.
- O simulador regula o agendamento de cada agente com base em seu estado: ATIVO, INATIVO, TERMINADO ou MORTO. Apenas agentes ativos estão autorizados a executar ações, e o simulador gerencia o tempo de execução alocado para cada agente; ao expirar, o agente é considerado MORTO.