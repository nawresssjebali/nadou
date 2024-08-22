import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CourierDetailsComponent } from './courier-details/courier-details.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,CourierDetailsComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'projet-nadou';
}
