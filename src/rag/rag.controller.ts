import { Controller, Get } from '@nestjs/common';
import { RagService } from './rag.service';

@Controller('rag')
export class RagController {
  constructor(private ragService: RagService) {}

  @Get('crag')
  async crag(): Promise<string> {
    return await this.ragService.executeCrag();
  }
}
