import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ConfigModule } from '@nestjs/config';
import { RagController } from './rag/rag.controller';
import { RagService } from './rag/rag.service';
@Module({
  imports: [
    ConfigModule.forRoot({
      envFilePath: ['.env'],
    }),
  ],
  controllers: [AppController, RagController],
  providers: [AppService, RagService],
})
export class AppModule {}
