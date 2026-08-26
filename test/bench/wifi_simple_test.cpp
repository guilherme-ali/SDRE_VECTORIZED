/**
 * Exemplo simples de teste de conexão WiFi/UDP
 * 
 * Este exemplo mostra como usar a biblioteca WiFiComm
 * para receber comandos do aplicativo ESP-Drone
 */

#include <Arduino.h>
#include <WiFiComm.h>

// Criar instância da comunicação WiFi
WiFiComm wifiComm("ESP-DRONE", "12345678", 2390);

// Variáveis para armazenar último comando
CommanderPacket lastCommand;
bool newCommand = false;

// Callback chamado quando um comando é recebido
void onCommandReceived(CommanderPacket cmd) {
    lastCommand = cmd;
    newCommand = true;
    
    Serial.println("\n🎮 NOVO COMANDO RECEBIDO:");
    Serial.printf("  Roll:   %+.3f\n", cmd.roll);
    Serial.printf("  Pitch:  %+.3f\n", cmd.pitch);
    Serial.printf("  Yaw:    %+.3f\n", cmd.yaw);
    Serial.printf("  Thrust: %d (%.1f%%)\n", cmd.thrust, (cmd.thrust / 60000.0f) * 100);
    Serial.println();
}

// Callback chamado quando cliente conecta
void onClientConnected() {
    Serial.println("\n✅ CLIENTE CONECTADO!");
    Serial.print("   IP: ");
    Serial.println(wifiComm.getClientIP());
    Serial.println();
}

// Callback chamado quando cliente desconecta
void onClientDisconnected() {
    Serial.println("\n❌ CLIENTE DESCONECTADO!");
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n╔════════════════════════════════════╗");
    Serial.println("║   TESTE DE CONEXÃO WiFi/UDP        ║");
    Serial.println("╚════════════════════════════════════╝\n");
    
    // Configurar debug
    wifiComm.enableDebug(true);    // Mostra conexões e comandos
    wifiComm.enableVerbose(false); // Não mostra todos os pacotes (para não poluir)
    
    // Inicializar WiFi
    if (wifiComm.begin()) {
        Serial.println("✅ Sistema WiFi/UDP inicializado!\n");
        
        // Registrar callbacks
        wifiComm.onCommandReceived(onCommandReceived);
        wifiComm.onClientConnected(onClientConnected);
        wifiComm.onClientDisconnected(onClientDisconnected);
        
        Serial.println("📱 Aguardando conexão do aplicativo...\n");
    } else {
        Serial.println("❌ ERRO ao inicializar WiFi/UDP!\n");
    }
}

void loop() {
    // IMPORTANTE: Sempre chamar update() no loop
    wifiComm.update();
    
    // Processar novo comando (se houver)
    if (newCommand) {
        newCommand = false;
        
        // Aqui você pode usar os valores do comando:
        // - lastCommand.roll
        // - lastCommand.pitch
        // - lastCommand.yaw
        // - lastCommand.thrust
        
        // Exemplo: Imprimir informações adicionais
        if (wifiComm.isClientConnected()) {
            // Fazer algo com o comando...
        }
    }
    
    // Pequeno delay para não sobrecarregar
    delay(1);
}
