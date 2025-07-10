#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <dlfcn.h>

#include "ocg_api.h"

// OCG Callback functions with correct signatures - marked extern "C" to prevent mangling
extern "C" {
    void card_reader(void* payload, uint32_t code, OCG_CardData* data) {
        std::cout << "Debug: card_reader called with code: " << code << std::endl;
        
        // Prevent recursive fetches on invalid codes
        if (code > 99999999 || code == 0) {
            std::cout << "Debug: Invalid code: " << code << ", skipping." << std::endl;
            return;
        }
        
        // Minimal card reader - just return basic card info
        if (data) {
            memset(data, 0, sizeof(OCG_CardData));
            data->code = code;
            data->alias = 0;
            data->setcodes = nullptr;
            data->type = 0x1; // TYPE_MONSTER
            data->level = 4;
            data->attribute = 1; // EARTH
            data->race = 1; // WARRIOR
            data->attack = 1000;
            data->defense = 1000;
            data->lscale = 0;
            data->rscale = 0;
            data->link_marker = 0;
            data->ot = 3; // OCG + TCG
        }
    }

    int script_reader(void* payload, OCG_Duel duel, const char* name) {
        std::cout << "Debug: script_reader called with name: " << (name ? name : "null") << std::endl;
        // Minimal script reader - always return 0 (script not found/not needed)
        return 0;
    }

    void log_handler(void* payload, const char* string, int type) {
        std::cout << "Debug: log_handler called: " << (string ? string : "null") << " (type " << type << ")" << std::endl;
    }

    void card_reader_done(void* payload, OCG_CardData* data) {
        std::cout << "Debug: card_reader_done called" << std::endl;
        // Minimal callback - do nothing
    }
}

// Function pointer types for dynamic loading
typedef int (*OCG_CreateDuel_t)(OCG_Duel*, const OCG_DuelOptions*);
typedef void (*OCG_DestroyDuel_t)(OCG_Duel);
typedef void (*OCG_DuelNewCard_t)(OCG_Duel, const OCG_NewCardInfo*);
typedef void (*OCG_StartDuel_t)(OCG_Duel);
typedef int (*OCG_DuelProcess_t)(OCG_Duel);
typedef void* (*OCG_DuelGetMessage_t)(OCG_Duel, uint32_t*);
typedef void (*OCG_DuelSetResponse_t)(OCG_Duel, const void*, uint32_t);
typedef int (*OCG_LoadScript_t)(OCG_Duel, const char*, uint32_t, const char*);

// Global function pointers
OCG_CreateDuel_t OCG_CreateDuel_ptr = nullptr;
OCG_DestroyDuel_t OCG_DestroyDuel_ptr = nullptr;
OCG_DuelNewCard_t OCG_DuelNewCard_ptr = nullptr;
OCG_StartDuel_t OCG_StartDuel_ptr = nullptr;
OCG_DuelProcess_t OCG_DuelProcess_ptr = nullptr;
OCG_DuelGetMessage_t OCG_DuelGetMessage_ptr = nullptr;
OCG_DuelSetResponse_t OCG_DuelSetResponse_ptr = nullptr;
OCG_LoadScript_t OCG_LoadScript_ptr = nullptr;

void* ocg_handle = nullptr;

struct Card {
    uint32_t code;
    std::string name;
};

struct Deck {
    std::vector<Card> main_deck;
    std::vector<Card> extra_deck;
    std::vector<Card> side_deck;
    std::string name;
};

class OCGHeadlessDuel {
private:
    OCG_Duel duel;
    bool initialized;
    int current_player;
    int winner;
    int turn_count;
    
public:
    OCGHeadlessDuel() : duel(nullptr), initialized(false), current_player(0), winner(-1), turn_count(0) {}
    
    ~OCGHeadlessDuel() {
        if (duel && OCG_DestroyDuel_ptr) {
            OCG_DestroyDuel_ptr(duel);
        }
        if (ocg_handle) {
            dlclose(ocg_handle);
        }
    }
    
    bool load_ocg_library(const std::string& lib_path) {
        ocg_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
        if (!ocg_handle) {
            std::cerr << "Failed to load OCG library: " << dlerror() << std::endl;
            return false;
        }
        
        // Load function pointers
        OCG_CreateDuel_ptr = (OCG_CreateDuel_t)dlsym(ocg_handle, "OCG_CreateDuel");
        OCG_StartDuel_ptr = (OCG_StartDuel_t)dlsym(ocg_handle, "OCG_StartDuel");
        OCG_DuelProcess_ptr = (OCG_DuelProcess_t)dlsym(ocg_handle, "OCG_DuelProcess");
        OCG_DuelNewCard_ptr = (OCG_DuelNewCard_t)dlsym(ocg_handle, "OCG_DuelNewCard");
        OCG_DuelGetMessage_ptr = (OCG_DuelGetMessage_t)dlsym(ocg_handle, "OCG_DuelGetMessage");
        OCG_DuelSetResponse_ptr = (OCG_DuelSetResponse_t)dlsym(ocg_handle, "OCG_DuelSetResponse");
        OCG_LoadScript_ptr = (OCG_LoadScript_t)dlsym(ocg_handle, "OCG_LoadScript");
        OCG_DestroyDuel_ptr = (OCG_DestroyDuel_t)dlsym(ocg_handle, "OCG_DestroyDuel");
        
        if (!OCG_CreateDuel_ptr || !OCG_StartDuel_ptr || !OCG_DuelProcess_ptr || 
            !OCG_DuelNewCard_ptr || !OCG_DuelGetMessage_ptr || !OCG_DuelSetResponse_ptr || 
            !OCG_DestroyDuel_ptr) {
            std::cerr << "Failed to load required OCG functions" << std::endl;
            return false;
        }
        
        return true;
    }
    
    Deck load_deck_from_ydk(const std::string& ydk_path) {
        Deck deck;
        std::ifstream file(ydk_path);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "Failed to open deck file: " << ydk_path << std::endl;
            return deck;
        }
        
        enum Section { MAIN, EXTRA, SIDE };
        Section current_section = MAIN;  // Default to main deck
        
        std::cout << "Debug: Loading deck from " << ydk_path << std::endl;
        
        while (std::getline(file, line)) {
            // Remove all whitespace from both ends
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue; // Empty line
            
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);
            
            if (line.empty()) continue;
            
            // Handle section markers
            if (line == "#main") {
                current_section = MAIN;
                std::cout << "Debug: Switching to MAIN section" << std::endl;
                continue;
            } else if (line == "#extra") {
                current_section = EXTRA;
                std::cout << "Debug: Switching to EXTRA section" << std::endl;
                continue;
            } else if (line == "!side") {
                current_section = SIDE;
                std::cout << "Debug: Switching to SIDE section" << std::endl;
                continue;
            }
            
            // Skip other comments
            if (line[0] == '#' || line[0] == '!') {
                std::cout << "Debug: Skipping comment/marker: " << line << std::endl;
                continue;
            }
            
            // Parse card code
            uint32_t card_code = 0;
            std::stringstream ss(line);
            if (ss >> card_code && card_code > 0) {
                Card card;
                card.code = card_code;
                card.name = "Card_" + std::to_string(card_code);
                
                switch (current_section) {
                    case MAIN:
                        deck.main_deck.push_back(card);
                        break;
                    case EXTRA:
                        deck.extra_deck.push_back(card);
                        break;
                    case SIDE:
                        deck.side_deck.push_back(card);
                        break;
                }
            }
        }
        
        std::cout << "Debug: Final deck stats - Main: " << deck.main_deck.size() 
                  << ", Extra: " << deck.extra_deck.size() 
                  << ", Side: " << deck.side_deck.size() << std::endl;
        
        return deck;
    }
    
    bool create_duel(const Deck& deck1, const Deck& deck2) {
        if (!OCG_CreateDuel_ptr) {
            std::cerr << "OCG library not loaded" << std::endl;
            return false;
        }
        
        // Create duel with proper OCG_DuelOptions struct
        OCG_DuelOptions options;
        memset(&options, 0, sizeof(options));
        
        std::cout << "Debug: OCG_DuelOptions struct size: " << sizeof(OCG_DuelOptions) << " bytes" << std::endl;
        
        // Set seed
        uint64_t seed = static_cast<uint64_t>(time(nullptr));
        options.seed[0] = seed;
        options.seed[1] = seed + 1;
        options.seed[2] = seed + 2;
        options.seed[3] = seed + 3;
        
        // Set flags (duel rules) - enable AI to automate responses
        options.flags = DUEL_TEST_MODE | DUEL_ATTACK_FIRST_TURN | DUEL_SIMPLE_AI;
        
        // Set team info
        options.team1.startingLP = 8000;
        options.team1.startingDrawCount = 5;
        options.team1.drawCountPerTurn = 1;
        
        options.team2.startingLP = 8000;
        options.team2.startingDrawCount = 5;
        options.team2.drawCountPerTurn = 1;
        
        // Set callbacks (all must be non-NULL according to documentation)
        // Create dummy payload pointers (pointing to simple integers)
        static int dummy_payload1 = 1;
        static int dummy_payload2 = 2;
        static int dummy_payload3 = 3;
        static int dummy_payload4 = 4;
        static int dummy_payload5 = 5;
        
        options.cardReader = card_reader;
        options.payload1 = &dummy_payload1;
        options.scriptReader = script_reader;
        options.payload2 = &dummy_payload2;
        options.logHandler = log_handler;
        options.payload3 = &dummy_payload3;
        options.cardReaderDone = card_reader_done;
        options.payload4 = &dummy_payload4;
        options.payload5 = &dummy_payload5;
        options.enableUnsafeLibraries = 0;
        // reserved array is already zeroed by memset
        
        std::cout << "Debug: Creating duel with options..." << std::endl;
        std::cout << "  sizeof(OCG_Player): " << sizeof(OCG_Player) << " bytes" << std::endl;
        std::cout << "  sizeof(OCG_DuelOptions): " << sizeof(OCG_DuelOptions) << " bytes" << std::endl;
        std::cout << "  struct size: " << sizeof(options) << " bytes" << std::endl;
        std::cout << "  seed[0]: " << options.seed[0] << std::endl;
        std::cout << "  flags: " << options.flags << std::endl;
        std::cout << "  team1 LP: " << options.team1.startingLP << std::endl;
        std::cout << "  team2 LP: " << options.team2.startingLP << std::endl;
        
        // Detailed callback pointer validation
        std::cout << "CardReader: " << (void*)options.cardReader << std::endl;
        std::cout << "ScriptReader: " << (void*)options.scriptReader << std::endl;
        std::cout << "LogHandler: " << (void*)options.logHandler << std::endl;
        std::cout << "CardReaderDone: " << (void*)options.cardReaderDone << std::endl;
        std::cout << "Payloads: "
                  << options.payload1 << ", "
                  << options.payload2 << ", "
                  << options.payload3 << ", "
                  << options.payload4 << ", "
                  << options.payload5 << std::endl;
        
        std::cout << "Debug: About to call OCG_CreateDuel..." << std::endl;
        
        // Create the duel (pass struct by pointer)
        std::cout << "Debug: Calling OCG_CreateDuel_ptr(" << &duel << ", " << &options << ")" << std::endl;
        int result = OCG_CreateDuel_ptr(&duel, &options);
        std::cout << "Debug: OCG_CreateDuel returned: " << result << std::endl;
        
        if (result != 0) {
            std::cerr << "Failed to create duel: " << result << std::endl;
            return false;
        }
        
        std::cout << "✅ Duel created successfully!" << std::endl;
        
        // Now add cards to the duel using OCG_DuelNewCard
        std::cout << "Adding cards to duel..." << std::endl;
        
        // Add player 1's main deck
        for (size_t i = 0; i < deck1.main_deck.size(); ++i) {
            OCG_NewCardInfo card_info;
            card_info.team = 0;
            card_info.duelist = 0;
            card_info.code = deck1.main_deck[i].code;
            card_info.con = 0;
            card_info.loc = LOCATION_DECK;
            card_info.seq = i;
            card_info.pos = 0;
            
            OCG_DuelNewCard_ptr(duel, &card_info);
        }
        
        // Add player 1's extra deck
        for (size_t i = 0; i < deck1.extra_deck.size(); ++i) {
            OCG_NewCardInfo card_info;
            card_info.team = 0;
            card_info.duelist = 0;
            card_info.code = deck1.extra_deck[i].code;
            card_info.con = 0;
            card_info.loc = LOCATION_EXTRA;
            card_info.seq = i;
            card_info.pos = 0;
            
            OCG_DuelNewCard_ptr(duel, &card_info);
        }
        
        // Add player 2's main deck
        for (size_t i = 0; i < deck2.main_deck.size(); ++i) {
            OCG_NewCardInfo card_info;
            card_info.team = 1;
            card_info.duelist = 0;
            card_info.code = deck2.main_deck[i].code;
            card_info.con = 1;
            card_info.loc = LOCATION_DECK;
            card_info.seq = i;
            card_info.pos = 0;
            
            OCG_DuelNewCard_ptr(duel, &card_info);
        }
        
        // Add player 2's extra deck
        for (size_t i = 0; i < deck2.extra_deck.size(); ++i) {
            OCG_NewCardInfo card_info;
            card_info.team = 1;
            card_info.duelist = 0;
            card_info.code = deck2.extra_deck[i].code;
            card_info.con = 1;
            card_info.loc = LOCATION_EXTRA;
            card_info.seq = i;
            card_info.pos = 0;
            
            OCG_DuelNewCard_ptr(duel, &card_info);
        }
        
        std::cout << "✅ All cards added to duel successfully!" << std::endl;
        
        initialized = true;
        return true;
    }
    
    bool start_duel() {
        if (!initialized || !OCG_StartDuel_ptr) {
            return false;
        }
        
        std::cout << "Starting duel..." << std::endl;
        OCG_StartDuel_ptr(duel);  // void function, no return value to check
        
        return true;
    }
    
    bool process_duel() {
        if (!initialized || !OCG_DuelProcess_ptr || !OCG_DuelGetMessage_ptr) {
            return false;
        }
        
        std::cout << "Processing duel..." << std::endl;
        
        int loop_count = 0;
        const int max_loops = 10000;  // Increase limit for complex duels
        
        // Process duel until it ends
        while (winner == -1 && loop_count < max_loops) {
            loop_count++;
            
            // Only show debug info every 100 loops to reduce spam
            if (loop_count % 100 == 1 || loop_count <= 10) {
                std::cout << "Debug: Loop " << loop_count << " - calling OCG_DuelProcess..." << std::endl;
            }
            
            int result = OCG_DuelProcess_ptr(duel);
            
            if (loop_count % 100 == 1 || loop_count <= 10) {
                std::cout << "Debug: OCG_DuelProcess returned: " << result << std::endl;
            }
            
            // Get and process messages
            uint32_t length = 0;
            void* message_buffer = OCG_DuelGetMessage_ptr(duel, &length);
            
            if (length > 0 && message_buffer) {
                process_message((uint8_t*)message_buffer, length);
            }
            
            if (result == 2) {  // OCG_DUEL_STATUS_END
                std::cout << "Debug: Duel process cycle ended (status 2)" << std::endl;
                // Don't break immediately - check if there are more messages or if winner is set
                if (length == 0 && winner == -1) {
                    // No more messages and no winner - duel might be truly over
                    std::cout << "Duel process ended - no more messages" << std::endl;
                    break;
                }
                continue;
            } else if (result == 1) {  // OCG_DUEL_STATUS_CONTINUE
                std::cout << "Duel continues (status 1)..." << std::endl;
                continue;
            } else if (result == 0) {  // OCG_DUEL_STATUS_AWAITING
                std::cout << "Duel waiting for response (status 0)..." << std::endl;
                
                // When status is 0, the duel is waiting for a response
                // Let's provide a minimal response and let the MSG_RETRY handler deal with retries
                if (OCG_DuelSetResponse_ptr) {
                    // Try just a single byte response indicating "proceed" or "no action"
                    uint8_t response = 0;
                    OCG_DuelSetResponse_ptr(duel, &response, 1);
                    std::cout << "Debug: Sent minimal response for AWAITING status" << std::endl;
                }
                continue;
            } else {
                std::cout << "Unknown duel status: " << result << std::endl;
                break;
            }
        }
        
        if (loop_count >= max_loops) {
            std::cout << "Warning: Reached maximum loop count, stopping duel processing" << std::endl;
        }
        
        return true;
    }
    
    bool process_message(uint8_t* buffer, uint32_t length) {
        if (length == 0) return true;
        
        uint8_t msg_type = buffer[0];
        std::cout << "Debug: Processing message type: " << (int)msg_type << " (length: " << length << ")" << std::endl;
        
        switch (msg_type) {
            case MSG_RETRY:
                std::cout << "🔄 MSG_RETRY - Engine requests retry (length: " << length << ")" << std::endl;
                // Print the full message to understand what's being requested
                if (length >= 5) {
                    std::cout << "MSG_RETRY data: ";
                    for (uint32_t i = 0; i < length; i++) {
                        std::cout << (int)buffer[i] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // The engine is asking for a choice - let's try to provide a valid choice response
                if (OCG_DuelSetResponse_ptr) {
                    // Based on the retry data pattern, try responding with choice 0 (first option)
                    uint8_t choice_response[] = {0};
                    OCG_DuelSetResponse_ptr(duel, choice_response, sizeof(choice_response));
                    std::cout << "Debug: Sent choice response (0)" << std::endl;
                }
                return true;
                
            case MSG_HINT:
                if (length >= 6) {
                    uint8_t player = buffer[1];
                    uint32_t hint_type = *(uint32_t*)(buffer + 2);
                    std::cout << "💡 Hint for Player " << (player + 1) << " (type: " << hint_type << ")" << std::endl;
                }
                return true;
                
            case MSG_WIN:
                if (length >= 2) {
                    winner = buffer[1];
                    std::cout << "🏆 Duel ended - Winner: Player " << (winner + 1) << std::endl;
                }
                return true;
                
            case MSG_NEW_TURN:
                if (length >= 2) {
                    current_player = buffer[1];
                    turn_count++;
                    std::cout << "🔄 Turn " << turn_count << " - Player " << (current_player + 1) << std::endl;
                }
                return true;
                
            case MSG_NEW_PHASE:
                if (length >= 2) {
                    uint8_t phase = buffer[1];
                    std::cout << "📋 New phase: " << (int)phase << std::endl;
                }
                return true;
                
            case MSG_DAMAGE:
                if (length >= 6) {
                    uint8_t player = buffer[1];
                    uint32_t damage = *(uint32_t*)(buffer + 2);
                    std::cout << "💥 Player " << (player + 1) << " takes " << damage << " damage" << std::endl;
                }
                return true;
                
            case MSG_LPUPDATE:
                if (length >= 6) {
                    uint8_t player = buffer[1];
                    uint32_t lp = *(uint32_t*)(buffer + 2);
                    std::cout << "❤️ Player " << (player + 1) << " LP: " << lp << std::endl;
                }
                return true;
                
            case MSG_DRAW:
                if (length >= 6) {
                    uint8_t player = buffer[1];
                    uint32_t count = *(uint32_t*)(buffer + 2);
                    std::cout << "🎴 Player " << (player + 1) << " draws " << count << " card(s)" << std::endl;
                }
                return true;
                
            case MSG_START:
                std::cout << "🚀 Duel started!" << std::endl;
                return true;
                
            case MSG_WAITING:
                std::cout << "⏳ Waiting..." << std::endl;
                return true;
                
            default:
                // Unknown message type, just continue
                std::cout << "Debug: Unknown message type: " << (int)msg_type << std::endl;
                return true;
        }
    }
    
    int get_winner() const { return winner; }
    int get_turn_count() const { return turn_count; }
    bool is_finished() const { return winner != -1; }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <deck1.ydk> <deck2.ydk> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --ocg-lib PATH    Path to libocgcore.so (default: /home/ecast229/Applications/EDOPro/libocgcore.so)" << std::endl;
    std::cout << "  --verbose         Enable verbose output" << std::endl;
    std::cout << "  --help            Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string deck1_path = argv[1];
    std::string deck2_path = argv[2];
    std::string ocg_lib_path = "/home/ecast229/Applications/EDOPro/libocgcore.so";
    bool verbose = false;
    
    // Parse arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ocg-lib" && i + 1 < argc) {
            ocg_lib_path = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "OCG Headless Duel Simulator" << std::endl;
    std::cout << "Deck 1: " << deck1_path << std::endl;
    std::cout << "Deck 2: " << deck2_path << std::endl;
    std::cout << "OCG Library: " << ocg_lib_path << std::endl;
    std::cout << "========================" << std::endl;
    
    OCGHeadlessDuel duel;
    
    // Load OCG library
    if (!duel.load_ocg_library(ocg_lib_path)) {
        std::cerr << "Failed to load OCG library" << std::endl;
        return 1;
    }
    
    // Load decks
    Deck deck1 = duel.load_deck_from_ydk(deck1_path);
    Deck deck2 = duel.load_deck_from_ydk(deck2_path);
    
    if (deck1.main_deck.empty() || deck2.main_deck.empty()) {
        std::cerr << "Failed to load decks or decks are empty" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded deck 1: " << deck1.main_deck.size() << " main, " 
              << deck1.extra_deck.size() << " extra cards" << std::endl;
    std::cout << "Loaded deck 2: " << deck2.main_deck.size() << " main, " 
              << deck2.extra_deck.size() << " extra cards" << std::endl;
    
    // Create and start duel
    if (!duel.create_duel(deck1, deck2)) {
        std::cerr << "Failed to create duel" << std::endl;
        return 1;
    }
    
    if (!duel.start_duel()) {
        std::cerr << "Failed to start duel" << std::endl;
        return 1;
    }
    
    std::cout << "Duel started!" << std::endl;
    
    // Process duel
    if (!duel.process_duel()) {
        std::cerr << "Error processing duel" << std::endl;
        return 1;
    }
    
    // Print final results
    std::cout << "========================" << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "Winner: Player " << (duel.get_winner() + 1) << std::endl;
    std::cout << "Total turns: " << duel.get_turn_count() << std::endl;
    
    return 0;
}
