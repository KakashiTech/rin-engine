/*
 * rin_distributed.h - Arquitectura de Nodos Distribuidos RIN
 * 
 * Preparado para futuro sistema Swarm Intelligence
 * Protocolo de resonancia de baja frecuencia entre nodos
 * 
 * NOTA: Esta es la infraestructura base. La comunicación inalámbrica
 * real requeriría hardware adicional (LoRa, WiFi HaLow, etc.)
 */

#ifndef RIN_DISTRIBUTED_H
#define RIN_DISTRIBUTED_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include "rin_core.h"
#include "rin_dptm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURACIÓN DE RED DISTRIBUIDA
 * ============================================================================ */

#define RIN_NODE_MAX_PEERS 16           /* Nodos conectados máximos */
#define RIN_NODE_MAX_PENDING 64         /* Mensajes pendientes máximos */
#define RIN_SWARM_SYNC_INTERVAL_MS 100  /* Intervalo de sincronización */
#define RIN_RESonance_FREQ_HZ 7.83f     /* Frecuencia Schumann (Hz) */

/* ============================================================================
 * TIPOS DE NODO
 * ============================================================================ */

typedef enum {
    RIN_NODE_TYPE_WORKER = 0,   /* Nodo de computo estándar */
    RIN_NODE_TYPE_COORDINATOR,  /* Nodo coordinador del swarm */
    RIN_NODE_TYPE_EDGE,         /* Nodo edge (entrada/salida) */
    RIN_NODE_TYPE_BRIDGE       /* Nodo puente a otros clusters */
} RIN_NodeType;

typedef enum {
    RIN_NODE_STATE_OFFLINE = 0,
    RIN_NODE_STATE_JOINING,
    RIN_NODE_STATE_ACTIVE,
    RIN_NODE_STATE_SYNCING,
    RIN_NODE_STATE_LEAVING
} RIN_NodeState;

/* ============================================================================
 * IDENTIDAD DE NODO
 * ============================================================================ */

typedef struct {
    uint64_t node_id;              /* ID único (hash de MAC/IP) */
    RIN_NodeType type;
    RIN_NodeState state;
    
    /* Capacidades */
    uint32_t compute_score;        /* Puntuación de capacidad (FLOPS estimados) */
    uint32_t memory_mb;            /* Memoria disponible */
    uint8_t energy_efficiency;     /* Eficiencia energética (0-255) */
    
    /* Ubicación (para optimización de topología) */
    float latency_ms;              /* Latencia al coordinador */
    float distance_hops;           /* Distancia en saltos */
    
    /* Timestamp */
    uint64_t last_seen_ms;
    uint64_t join_time_ms;
} RIN_NodeInfo;

/* ============================================================================
 * MENSAJES DE PROTOCOLO
 * ============================================================================ */

typedef enum {
    RIN_MSG_PING = 0,        /* Heartbeat */
    RIN_MSG_PONG,            /* Respuesta heartbeat */
    RIN_MSG_JOIN,            /* Solicitud unirse a swarm */
    RIN_MSG_JOIN_ACK,        /* Confirmación unión */
    RIN_MSG_LEAVE,           /* Notificación salida */
    RIN_MSG_SYNC,            /* Sincronización de estado */
    RIN_MSG_INFERENCE_REQ,   /* Solicitud de inferencia */
    RIN_MSG_INFERENCE_RESP,  /* Respuesta de inferencia */
    RIN_MSG_GRADIENT_SHARE,  /* Compartir gradientes (entrenamiento distribuido) */
    RIN_MSG_WEIGHT_SYNC,     /* Sincronización de pesos */
    RIN_MSG_RESONANCE_PULSE  /* Pulso de resonancia (sincronización temporal) */
} RIN_MessageType;

/* Header de mensaje */
typedef struct {
    RIN_MessageType type;
    uint64_t src_node_id;
    uint64_t dst_node_id;     /* 0 = broadcast */
    uint32_t seq_num;
    uint32_t payload_len;
    uint64_t timestamp_ms;
    uint32_t checksum;
} RIN_MessageHeader;

/* Mensaje completo */
typedef struct {
    RIN_MessageHeader header;
    uint8_t* payload;
    uint32_t payload_capacity;
} RIN_Message;

/* ============================================================================
 * PULSO DE RESONANCIA (Sincronización Temporal)
 * ============================================================================ */

typedef struct {
    uint64_t pulse_id;         /* ID de secuencia del pulso */
    uint64_t origin_time_ns;   /* Tiempo de origen (nanosegundos) */
    uint64_t ttl_ms;           /* Time-to-live */
    float frequency_hz;        /* Frecuencia de resonancia (Schumann ~7.83Hz) */
    uint8_t phase_offset;      /* Offset de fase (0-255 = 0-2π) */
    uint32_t hop_count;        /* Contador de saltos */
} RIN_ResonancePulse;

/* ============================================================================
 * SHARD DE MODELO (Distribución del modelo entre nodos)
 * ============================================================================ */

typedef struct {
    uint32_t shard_id;         /* ID de este shard */
    uint32_t total_shards;     /* Total de shards en el swarm */
    uint32_t layer_start;      /* Capa inicial */
    uint32_t layer_end;        /* Capa final */
    uint64_t node_id;          /* Nodo responsable */
} RIN_ModelShard;

/* ============================================================================
 * ESTADÍSTICAS DE SWARM
 * ============================================================================ */

typedef struct {
    uint32_t total_nodes;
    uint32_t active_nodes;
    uint32_t total_params;     /* Parámetros totales del modelo distribuido */
    double total_flops;        /* FLOPS agregados del swarm */
    double avg_latency_ms;
    float sync_coherence;      /* Coherencia de fase (0-1) */
    uint64_t messages_sent;
    uint64_t messages_recv;
    uint64_t bytes_transferred;
    double energy_total_joules; /* Energía total consumida por el swarm */
} RIN_SwarmStats;

/* ============================================================================
 * CONTEXTO DE NODO DISTRIBUIDO
 * ============================================================================ */

typedef struct {
    /* Nodo local */
    RIN_NodeInfo self;
    RIN_Context* local_model;  /* Modelo local (puede ser shard) */
    
    /* Peers conectados */
    RIN_NodeInfo peers[RIN_NODE_MAX_PEERS];
    uint32_t num_peers;
    
    /* Cola de mensajes */
    RIN_Message message_queue[RIN_NODE_MAX_PENDING];
    uint32_t msg_head;
    uint32_t msg_tail;
    
    /* Sharding del modelo */
    RIN_ModelShard my_shard;
    RIN_ModelShard* all_shards;
    uint32_t num_shards;
    
    /* Estado del swarm */
    RIN_SwarmStats stats;
    uint64_t last_sync_ms;
    
    /* Callbacks de transporte (plataforma específica) */
    int (*transport_send)(uint64_t dst_id, const uint8_t* data, uint32_t len);
    int (*transport_recv)(uint64_t* src_id, uint8_t* buffer, uint32_t max_len);
    int (*transport_broadcast)(const uint8_t* data, uint32_t len);
    
    /* Estado de ejecución */
    bool running;
    uint64_t start_time_ms;
    
} RIN_DistributedContext;

/* ============================================================================
 * FUNCIONES DE INICIALIZACIÓN
 * ============================================================================ */

/*
 * RIN_Distributed_Init - Inicializar nodo distribuido
 * 
 * @ctx: Contexto a inicializar
 * @node_type: Tipo de este nodo
 * @local_model: Modelo local (puede ser NULL si solo es relay)
 * 
 * Retorna: 0 si éxito
 */
static inline int RIN_Distributed_Init(RIN_DistributedContext* ctx,
                                         RIN_NodeType node_type,
                                         RIN_Context* local_model) {
    if (!ctx) return -1;
    
    memset(ctx, 0, sizeof(RIN_DistributedContext));
    
    /* Generar ID único (simplificado - en producción usar MAC address) */
    ctx->self.node_id = (uint64_t)time(NULL) ^ (uint64_t)getpid();
    ctx->self.type = node_type;
    ctx->self.state = RIN_NODE_STATE_OFFLINE;
    ctx->self.join_time_ms = RIN_DPTM_GetTimestampMs();
    
    if (local_model) {
        ctx->local_model = local_model;
        RIN_ModelStats stats;
        RIN_GetModelStats(local_model, &stats);
        ctx->self.compute_score = stats.num_parameters / 1000;  /* Estimación simple */
        ctx->self.memory_mb = stats.weights_size_bytes / (1024*1024);
    }
    
    ctx->num_peers = 0;
    ctx->msg_head = 0;
    ctx->msg_tail = 0;
    ctx->running = false;
    ctx->start_time_ms = RIN_DPTM_GetTimestampMs();
    
    /* Inicializar shard local como todo el modelo por defecto */
    ctx->my_shard.shard_id = 0;
    ctx->my_shard.total_shards = 1;
    ctx->my_shard.layer_start = 0;
    if (local_model) {
        ctx->my_shard.layer_end = local_model->num_layers;
    } else {
        ctx->my_shard.layer_end = 0;
    }
    ctx->my_shard.node_id = ctx->self.node_id;
    
    return 0;
}

/*
 * RIN_Distributed_SetTransport - Configurar callbacks de transporte
 * 
 * Los callbacks deben ser implementados por la plataforma:
 * - WiFi, LoRa, Ethernet, etc.
 */
static inline void RIN_Distributed_SetTransport(
    RIN_DistributedContext* ctx,
    int (*send_fn)(uint64_t, const uint8_t*, uint32_t),
    int (*recv_fn)(uint64_t*, uint8_t*, uint32_t),
    int (*broadcast_fn)(const uint8_t*, uint32_t)) {
    
    if (!ctx) return;
    
    ctx->transport_send = send_fn;
    ctx->transport_recv = recv_fn;
    ctx->transport_broadcast = broadcast_fn;
}

/* ============================================================================
 * FUNCIONES DE PROTOCOLO
 * ============================================================================ */

/*
 * RIN_Distributed_JoinSwarm - Unirse a un swarm existente
 * 
 * Enviar mensaje JOIN a coordinador conocido o broadcast
 */
static inline int RIN_Distributed_JoinSwarm(RIN_DistributedContext* ctx,
                                             uint64_t coordinator_id) {
    if (!ctx || !ctx->transport_send) return -1;
    
    ctx->self.state = RIN_NODE_STATE_JOINING;
    
    /* Crear mensaje JOIN */
    RIN_Message msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.type = RIN_MSG_JOIN;
    msg.header.src_node_id = ctx->self.node_id;
    msg.header.dst_node_id = coordinator_id;
    msg.header.seq_num = 0;
    msg.header.timestamp_ms = RIN_DPTM_GetTimestampMs();
    
    /* Payload: info del nodo */
    uint8_t payload[64];
    memcpy(payload, &ctx->self, sizeof(RIN_NodeInfo));
    msg.payload = payload;
    msg.header.payload_len = sizeof(RIN_NodeInfo);
    
    /* Enviar */
    if (coordinator_id == 0) {
        /* Broadcast */
        if (ctx->transport_broadcast) {
            ctx->transport_broadcast((uint8_t*)&msg.header, sizeof(msg.header));
        }
    } else {
        /* Unicast */
        ctx->transport_send(coordinator_id, (uint8_t*)&msg.header, sizeof(msg.header));
    }
    
    return 0;
}

/*
 * RIN_Distributed_SendResonancePulse - Enviar pulso de sincronización
 * 
 * Basado en resonancia de Schumann (7.83 Hz) para sincronización temporal
 */
static inline int RIN_Distributed_SendResonancePulse(RIN_DistributedContext* ctx) {
    if (!ctx) return -1;
    
    RIN_ResonancePulse pulse;
    pulse.pulse_id = ctx->stats.messages_sent++;
    pulse.origin_time_ns = RIN_DPTM_GetTimestampNs();
    pulse.ttl_ms = 5000;  /* 5 segundos TTL */
    pulse.frequency_hz = RIN_RESonance_FREQ_HZ;
    pulse.phase_offset = (uint8_t)((pulse.origin_time_ns / 1000000ULL) % 256);
    pulse.hop_count = 0;
    
    /* Crear mensaje */
    RIN_MessageHeader header;
    header.type = RIN_MSG_RESONANCE_PULSE;
    header.src_node_id = ctx->self.node_id;
    header.dst_node_id = 0;  /* Broadcast */
    header.seq_num = pulse.pulse_id;
    header.timestamp_ms = RIN_DPTM_GetTimestampMs();
    header.payload_len = sizeof(RIN_ResonancePulse);
    
    /* Enviar si hay transporte */
    if (ctx->transport_broadcast) {
        ctx->transport_broadcast((uint8_t*)&header, sizeof(header));
        ctx->transport_broadcast((uint8_t*)&pulse, sizeof(pulse));
    }
    
    return 0;
}

/*
 * RIN_Distributed_ProcessMessage - Procesar mensaje recibido
 * 
 * Máquina de estados del protocolo
 */
static inline void RIN_Distributed_ProcessMessage(RIN_DistributedContext* ctx,
                                                 const RIN_Message* msg) {
    if (!ctx || !msg) return;
    
    switch (msg->header.type) {
        case RIN_MSG_PING:
            /* Responder con PONG */
            break;
            
        case RIN_MSG_JOIN:
            /* Añadir nuevo peer */
            if (ctx->num_peers < RIN_NODE_MAX_PEERS) {
                memcpy(&ctx->peers[ctx->num_peers], msg->payload, sizeof(RIN_NodeInfo));
                ctx->peers[ctx->num_peers].node_id = msg->header.src_node_id;
                ctx->peers[ctx->num_peers].last_seen_ms = RIN_DPTM_GetTimestampMs();
                ctx->num_peers++;
                ctx->stats.active_nodes++;
            }
            break;
            
        case RIN_MSG_SYNC:
            /* Sincronizar estado del modelo */
            break;
            
        case RIN_MSG_INFERENCE_REQ:
            /* Procesar solicitud de inferencia */
            break;
            
        case RIN_MSG_RESONANCE_PULSE:
            /* Actualizar sincronización de fase */
            if (msg->payload && msg->header.payload_len >= sizeof(RIN_ResonancePulse)) {
                RIN_ResonancePulse* pulse = (RIN_ResonancePulse*)msg->payload;
                /* Calcular coherencia de fase */
                uint64_t delay_ns = RIN_DPTM_GetTimestampNs() - pulse->origin_time_ns;
                float delay_ms = delay_ns / 1000000.0f;
                ctx->stats.avg_latency_ms = (ctx->stats.avg_latency_ms + delay_ms) / 2.0f;
            }
            break;
            
        default:
            break;
    }
    
    ctx->stats.messages_recv++;
}

/* ============================================================================
 * INFERENCIA DISTRIBUIDA
 * ============================================================================ */

/*
 * RIN_Distributed_Inference - Inferencia distribuida en el swarm
 * 
 * El token fluye a través de los shards del modelo
 * Cada nodo procesa sus capas y pasa al siguiente
 */
static inline int RIN_Distributed_Inference(RIN_DistributedContext* ctx,
                                           const uint32_t* input_ids,
                                           uint32_t num_input,
                                           uint32_t* output_ids,
                                           uint32_t max_output) {
    if (!ctx) return -1;
    
    /* Si somos el único nodo, usar inferencia local */
    if (ctx->num_peers == 0 || ctx->num_shards <= 1) {
        if (ctx->local_model) {
            RIN_InferenceResult result;
            return RIN_Inference(ctx->local_model, input_ids, num_input, max_output, &result);
        }
        return -1;
    }
    
    /* TODO: Implementar pipeline distribuido */
    /* Por ahora, fallback a local */
    
    return -1;
}

/* ============================================================================
 * SHARDING DE MODELO
 * ============================================================================ */

/*
 * RIN_Distributed_ConfigureSharding - Configurar sharding del modelo
 * 
 * Divide el modelo entre múltiples nodos
 */
static inline int RIN_Distributed_ConfigureSharding(RIN_DistributedContext* ctx,
                                                 uint32_t num_shards) {
    if (!ctx || num_shards == 0 || num_shards > RIN_NODE_MAX_PEERS) return -1;
    
    ctx->num_shards = num_shards;
    ctx->all_shards = (RIN_ModelShard*)calloc(num_shards, sizeof(RIN_ModelShard));
    
    if (!ctx->local_model) return -1;
    
    uint32_t layers_per_shard = ctx->local_model->num_layers / num_shards;
    
    for (uint32_t i = 0; i < num_shards; i++) {
        ctx->all_shards[i].shard_id = i;
        ctx->all_shards[i].total_shards = num_shards;
        ctx->all_shards[i].layer_start = i * layers_per_shard;
        ctx->all_shards[i].layer_end = (i + 1) * layers_per_shard;
        if (i == num_shards - 1) {
            /* Último shard toma capas restantes */
            ctx->all_shards[i].layer_end = ctx->local_model->num_layers;
        }
    }
    
    /* Asignar shard local */
    /* Buscar nuestro node_id en la lista de shards */
    for (uint32_t i = 0; i < num_shards; i++) {
        if (ctx->all_shards[i].node_id == ctx->self.node_id) {
            ctx->my_shard = ctx->all_shards[i];
            break;
        }
    }
    
    return 0;
}

/* ============================================================================
 * ESTADÍSTICAS Y MONITORING
 * ============================================================================ */

/*
 * RIN_Distributed_GetStats - Obtener estadísticas del swarm
 */
static inline void RIN_Distributed_GetStats(const RIN_DistributedContext* ctx,
                                           RIN_SwarmStats* stats) {
    if (!ctx || !stats) return;
    *stats = ctx->stats;
}

/*
 * RIN_Distributed_PrintStatus - Imprimir estado del nodo
 */
static inline void RIN_Distributed_PrintStatus(const RIN_DistributedContext* ctx) {
    if (!ctx) return;
    
    printf("\n========== RIN Distributed Node Status ==========\n");
    printf("Node ID: 0x%016lX\n", ctx->self.node_id);
    printf("Type: %s\n", 
           ctx->self.type == RIN_NODE_TYPE_COORDINATOR ? "Coordinator" :
           ctx->self.type == RIN_NODE_TYPE_WORKER ? "Worker" :
           ctx->self.type == RIN_NODE_TYPE_EDGE ? "Edge" : "Bridge");
    printf("State: %s\n",
           ctx->self.state == RIN_NODE_STATE_ACTIVE ? "Active" :
           ctx->self.state == RIN_NODE_STATE_JOINING ? "Joining" :
           ctx->self.state == RIN_NODE_STATE_OFFLINE ? "Offline" : "Other");
    printf("Peers connected: %u/%d\n", ctx->num_peers, RIN_NODE_MAX_PEERS);
    printf("Model shard: %u/%u (layers %u-%u)\n",
           ctx->my_shard.shard_id, ctx->my_shard.total_shards,
           ctx->my_shard.layer_start, ctx->my_shard.layer_end);
    printf("Swarm stats: %u nodes, %.2f ms avg latency\n",
           ctx->stats.active_nodes, ctx->stats.avg_latency_ms);
    printf("Messages: %lu sent, %lu recv\n",
           ctx->stats.messages_sent, ctx->stats.messages_recv);
    printf("================================================\n\n");
}

/* ============================================================================
 * LIMPIEZA
 * ============================================================================ */

static inline void RIN_Distributed_Destroy(RIN_DistributedContext* ctx) {
    if (!ctx) return;
    
    /* Notificar salida */
    if (ctx->self.state == RIN_NODE_STATE_ACTIVE && ctx->transport_broadcast) {
        RIN_MessageHeader header;
        header.type = RIN_MSG_LEAVE;
        header.src_node_id = ctx->self.node_id;
        header.dst_node_id = 0;
        header.seq_num = 0;
        header.timestamp_ms = RIN_DPTM_GetTimestampMs();
        header.payload_len = 0;
        
        ctx->transport_broadcast((uint8_t*)&header, sizeof(header));
    }
    
    /* Liberar recursos */
    if (ctx->all_shards) {
        free(ctx->all_shards);
        ctx->all_shards = NULL;
    }
    
    ctx->running = false;
}

#ifdef __cplusplus
}
#endif

#endif /* RIN_DISTRIBUTED_H */
