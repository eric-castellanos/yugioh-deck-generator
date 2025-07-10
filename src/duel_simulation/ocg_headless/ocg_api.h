#ifndef OCG_API_H
#define OCG_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// OCG API Types
typedef void* OCG_Duel;

// Player structure
#pragma pack(push, 1)
typedef struct {
    uint32_t startingLP;
    uint32_t startingDrawCount;
    uint32_t drawCountPerTurn;
} OCG_Player;
#pragma pack(pop)

// Card data structure
typedef struct {
    uint32_t code;
    uint32_t alias;
    uint16_t* setcodes;
    uint32_t type;
    int32_t level;
    uint32_t attribute;
    uint64_t race;
    int32_t attack;
    int32_t defense;
    uint32_t lscale;
    uint32_t rscale;
    uint32_t link_marker;
    uint32_t ot;
} OCG_CardData;

// New card info structure
typedef struct {
    uint8_t team;
    uint8_t duelist;
    uint32_t code;
    uint8_t con;
    uint32_t loc;
    uint32_t seq;
    uint32_t pos;
} OCG_NewCardInfo;

// Callback function types
typedef void (*OCG_DataReader)(void* payload, uint32_t code, OCG_CardData* data);
typedef int (*OCG_ScriptReader)(void* payload, OCG_Duel duel, const char* name);
typedef void (*OCG_LogHandler)(void* payload, const char* string, int type);
typedef void (*OCG_DataReaderDone)(void* payload, OCG_CardData* data);

// OCG Duel Options struct (correct layout - callbacks first!)
#pragma pack(push, 1)
typedef struct OCG_DuelOptions {
    OCG_DataReader cardReader;          // 8 bytes - Card data callback
    void* payload1;                     // 8 bytes - cardReader payload
    OCG_ScriptReader scriptReader;      // 8 bytes - Script loader callback
    void* payload2;                     // 8 bytes - scriptReader payload
    OCG_LogHandler logHandler;          // 8 bytes - Log/message callback
    void* payload3;                     // 8 bytes - logHandler payload
    OCG_DataReaderDone cardReaderDone;  // 8 bytes - Card done callback
    void* payload4;                     // 8 bytes - cardReaderDone payload
    void* payload5;                     // 8 bytes - Additional payload (kdiy extension)
    uint64_t seed[4];                   // 32 bytes - RNG seed array
    uint64_t flags;                     // 8 bytes - Duel mode flags
    OCG_Player team1;                   // 12 bytes - Player 1 config
    OCG_Player team2;                   // 12 bytes - Player 2 config
    uint8_t enableUnsafeLibraries;      // 1 byte - Enable unsafe Lua libs
    uint8_t _padding[23];               // 23 bytes - Padding to reach 160 bytes total
} OCG_DuelOptions;
#pragma pack(pop)

// OCG Constants
#define OCG_PLAYER_NONE 2
#define OCG_REASON_RULE 0
#define OCG_REASON_COST 1
#define OCG_REASON_EFFECT 2

// Duel options flags (from ocgcore source)
#define DUEL_TEST_MODE         0x01
#define DUEL_ATTACK_FIRST_TURN 0x02
#define DUEL_USE_TRAPS_IN_NEW_CHAIN 0x04
#define DUEL_6_STEP_BATLLE_STEP 0x08
#define DUEL_PSEUDO_SHUFFLE    0x10
#define DUEL_TRIGGER_WHEN_PRIVATE_KNOWLEDGE 0x20
#define DUEL_SIMPLE_AI         0x40

// Message types
#define MSG_RETRY 1
#define MSG_HINT 2
#define MSG_WIN 5
#define MSG_WAITING 6
#define MSG_START 7
#define MSG_UPDATE_DATA 8
#define MSG_DRAW 60
#define MSG_SHUFFLEDECK 61
#define MSG_SHUFFLE_HAND 62
#define MSG_CHAINING 70
#define MSG_CHAINED 71
#define MSG_DAMAGE 91
#define MSG_RECOVER 92
#define MSG_LPUPDATE 93
#define MSG_NEW_TURN 40
#define MSG_NEW_PHASE 41

// Query types
#define QUERY_CODE 0x1
#define QUERY_POSITION 0x2
#define QUERY_ALIAS 0x4
#define QUERY_TYPE 0x8
#define QUERY_LEVEL 0x10
#define QUERY_RANK 0x20
#define QUERY_ATTRIBUTE 0x40
#define QUERY_RACE 0x80
#define QUERY_ATTACK 0x100
#define QUERY_DEFENSE 0x200
#define QUERY_BASE_ATTACK 0x400
#define QUERY_BASE_DEFENSE 0x800
#define QUERY_REASON 0x1000
#define QUERY_OWNER 0x4000
#define QUERY_STATUS 0x8000
#define QUERY_LSCALE 0x10000
#define QUERY_RSCALE 0x20000
#define QUERY_LINK 0x40000

// Card locations
#define LOCATION_DECK 0x01
#define LOCATION_HAND 0x02
#define LOCATION_MZONE 0x04
#define LOCATION_SZONE 0x08
#define LOCATION_GRAVE 0x10
#define LOCATION_REMOVED 0x20
#define LOCATION_EXTRA 0x40
#define LOCATION_OVERLAY 0x80

// OCG API Functions
int OCG_CreateDuel(OCG_Duel* duel, const OCG_DuelOptions* options);
void OCG_DestroyDuel(OCG_Duel duel);
void OCG_DuelNewCard(OCG_Duel duel, const OCG_NewCardInfo* info);
void OCG_StartDuel(OCG_Duel duel);
int OCG_DuelProcess(OCG_Duel duel);
void* OCG_DuelGetMessage(OCG_Duel duel, uint32_t* length);
void OCG_DuelSetResponse(OCG_Duel duel, const void* buffer, uint32_t length);
int OCG_LoadScript(OCG_Duel duel, const char* buffer, uint32_t length, const char* name);

#ifdef __cplusplus
}
#endif

#endif // OCG_API_H
