# SARA Simple Planner (ROS1)

Nodo ligero en Python para ejecutar una trayectoria cerrada por waypoints
utilizando como realimentación una pose estimada por el sistema de
localización.

## Ruta incluida

Se han seleccionado cuatro landmarks del mapa suministrado porque forman
un cuadrilátero casi rectangular:

- P1 -> ArUco 3  : (1.4650, 0.0000)
- P2 -> ArUco 41 : (2.8899, -0.0154)
- P3 -> ArUco 25 : (2.8486, 2.6443)
- P4 -> ArUco 19 : (1.4284, 2.7621)

La ruta experimental es:

P1 -> P2 -> P3 -> P4 -> P1

y por defecto se repite 5 veces.

Los ángulos de los waypoints están orientados aproximadamente hacia el
siguiente lado del polígono.

## Instalación

Copiar la carpeta `sara_simple_planner` dentro de `~/catkin_ws/src/`.

Después:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Aunque el nodo es Python, `catkin_make` registra el paquete y sus
dependencias en el workspace.

## Antes de arrancar

Comprobar qué topic publica la pose del filtro:

```bash
rostopic list | grep pose
```

El launch usa por defecto:

```text
/update_pose
```

Debe ser la pose FUSIONADA por el filtro, no la odometría `/pose`.

Si posteriormente la salida se llama `/localization/inc_ukf`:

```bash
roslaunch sara_simple_planner simple_planner.launch \
    pose_topic:=/localization/inc_ukf
```

Comprobar también el topic real de velocidad de la silla:

```bash
rostopic info /cmd_vel
```

## Arranque

```bash
roslaunch sara_simple_planner simple_planner.launch
```

Por seguridad `auto_start=false`: lanzar el nodo NO mueve la silla.

Cuando la pose se esté recibiendo correctamente:

```bash
rosservice call /simple_planner/start
```

Parada:

```bash
rosservice call /simple_planner/stop
```

## Modo de medición manual

Para la campaña de ground truth, cambiar en el launch:

```xml
<param name="manual_checkpoint" value="true"/>
```

En este modo la silla se detiene indefinidamente en cada checkpoint.

Tras realizar la medida física:

```bash
rosservice call /simple_planner/continue
```

## Topics

Entrada:

```text
/update_pose                  geometry_msgs/Pose2D
```

Salida:

```text
/cmd_vel                      geometry_msgs/Twist
/simple_planner/target_pose   geometry_msgs/Pose2D
/simple_planner/checkpoint_reached std_msgs/String
```

## Controlador

Primera fase:

```text
GO_TO_POSITION
```

El controlador utiliza distancia al waypoint y error angular hacia el
waypoint.

Segunda fase:

```text
ALIGN_HEADING
```

Una vez alcanzada la posición, la silla rota sobre sí misma hasta alcanzar
el `theta` definido para ese checkpoint.

Después entra en:

```text
CHECKPOINT
```

y permanece parada durante `dwell_time` o, si `manual_checkpoint=true`,
hasta recibir `/simple_planner/continue`.

## Seguridad

Los límites por defecto son deliberadamente conservadores:

```text
max_linear_velocity  = 0.25 m/s
max_angular_velocity = 0.45 rad/s
```

Además:
- si el error angular supera 0.70 rad, la velocidad lineal se fuerza a cero;
- si la pose deja de recibirse durante más de 0.5 s, se publica velocidad cero;
- al apagar el nodo se publica `Twist()` nulo;
- el nodo no comienza a moverse automáticamente por defecto.

Aun así, validar primero con las ruedas levantadas o con una velocidad
máxima muy reducida antes de ejecutar la trayectoria completa.

## Log

Los eventos se guardan por defecto en:

```text
/tmp/simple_planner_events.csv
```

Incluye:
- timestamp,
- loop,
- waypoint,
- ArUco asociado,
- referencia `(x,y,theta)`,
- pose estimada cuando se alcanza el checkpoint.

Este log NO es ground truth físico; la pose que almacena es únicamente la
pose de feedback utilizada por el controlador.
